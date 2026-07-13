!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
! File lsmrReverse.f90
!
!     LSMR
!
! LSMR   solves Ax = b or min ||Ax - b|| with or without damping,
! using the iterative algorithm of David Fong and Michael Saunders:
!     http://www.stanford.edu/group/SOL/software/lsmr.html
!
! The original LSMR code can be found at
! http://web.stanford.edu/group/SOL/software/lsmr/
! It is maintained by
!     David Fong       <clfong@stanford.edu>
!     Michael Saunders <saunders@stanford.edu>
!     Systems Optimization Laboratory (SOL)
!     Stanford University
!     Stanford, CA 94305-4026, USA

! This version uses reverse communication so that control is passed back
! to the user to perform products with A or A'.
! In addition, the user may choose to use their own stopping rule(s)
! or to apply the stopping criteria of Fong and Saunders.
! A number of options are available (see the derived type lsmr_type_options).
!
! 17 Jul 2010: F90 LSMR derived from F90 LSQR and lsqr.m.
!              Aprod1, Aprod2 implemented via f90 interface.
! 07 Sep 2010: Local reorthogonalization now works (localSize > 0).
! 28 Jan 2014: In lsmrDataModule.f90:
!              ip added for integer(ip) declarations.
!              dnrm2 and dscal coded directly
!              (no longer use lsmrblasInterface.f90 or lsmrblas.f90).
! 02 May 2014: With damp>0, istop=2 was incorrectly set to istop=3
!              (so incorrect stopping message was printed).  Fixed.
! 20 May 2014: Initial reverse-communication version written by Nick Gould
! 23 Nov 2015: Revised reverse communication interface written by
!              Nick Gould and Jennifer Scott
!              nick.gould@stfc.ac.uk, jennifer.scott@stfc.ac.uk
!              STFC Rutherford Appleton Laboratory
!              Harwell, Oxford, Didcot
!              United Kingdom OX11 0QX
!
!              lsmrDataModule.f90 no longer used.
!              Automatic arrays no longer used.
!              Option for user to test convergence (or to use Fong and 
!              Saunders test).
!              Uses dnrm2 and dscal (for closer compatibility with 
!              Fong and Saunders code ... easier to compare results).
!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    ! Precision
    ! ---------
    ! The number of iterations required by LSMR will decrease
    ! if the computation is performed in higher precision.
    ! At least 15-digit arithmetic should normally be used.
    ! "real(dp)" declarations should normally be 8-byte words.

!+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

module lsmrReverse

  implicit none

  integer(4),  parameter :: ip = kind( 1 )
  integer(4),  parameter :: dp = kind( 1.0d+0 )
  real(dp),    parameter :: zero = 0.0_dp, one = 1.0_dp, onept = 1.1_dp

  private
  public   :: LSMR_reverse

    !-------------------------------------------------------------------
    ! LSMR  finds a solution x to the following problems:
    !
    ! 1. Unsymmetric equations:    Solve  A*x = b
    !
    ! 2. Linear least squares:     Solve  A*x = b
    !                              in the least-squares sense
    !
    ! 3. Damped least squares:     Solve  (   A    )*x = ( b )
    !                                     ( damp*I )     ( 0 )
    !                              in the least-squares sense
    !
    ! where A is a matrix with m rows and n columns, b is an m-vector,
    ! and damp is a scalar.  (All quantities are real.)
    ! The matrix A is treated as a linear operator.  It is accessed
    ! by reverse communication, that is requests for matrix-vector 
    ! products are passed back to the user.
    !
    ! LSMR uses an iterative method to approximate the solution.
    ! The number of iterations required to reach a certain accuracy
    ! depends strongly on the scaling of the problem.  Poor scaling of
    ! the rows or columns of A should therefore be avoided where
    ! possible.
    !
    ! For example, in problem 1 the solution is unaltered by
    ! row-scaling.  If a row of A is very small or large compared to
    ! the other rows of A, the corresponding row of ( A  b ) should be
    ! scaled up or down.
    !
    ! In problems 1 and 2, the solution x is easily recovered
    ! following column-scaling.  Unless better information is known,
    ! the nonzero columns of A should be scaled so that they all have
    ! the same Euclidean norm (e.g., 1.0).
    !
    ! In problem 3, there is no freedom to re-scale if damp is
    ! nonzero.  However, the value of damp should be assigned only
    ! after attention has been paid to the scaling of A.
    !
    ! The parameter damp is intended to help regularize
    ! ill-conditioned systems, by preventing the true solution from
    ! being very large.  Another aid to regularization is provided by
    ! the parameter condA, which may be used to terminate iterations
    ! before the computed solution becomes very large.
    !
    ! Note that x is not an input parameter.
    ! If some initial estimate x0 is known and if damp = 0,
    ! one could proceed as follows:
    !
    ! 1. Compute a residual vector     r0 = b - A*x0.
    ! 2. Use LSMR to solve the system  A*dx = r0.
    ! 3. Add the correction dx to obtain a final solution x = x0 + dx.
    !
    ! This requires that x0 be available before the first call 
    ! to LSMR_reverse and after the final call. 
    ! To judge the benefits, suppose LSMR takes k1 iterations
    ! to solve A*x = b and k2 iterations to solve A*dx = r0.
    ! If x0 is "good", norm(r0) will be smaller than norm(b).
    ! If the same stopping tolerances atol and btol are used for each
    ! system, k1 and k2 will be similar, but the final solution x0 + dx
    ! should be more accurate.  The only way to reduce the total work
    ! is to use a larger stopping tolerance for the second system.
    ! If some value btol is suitable for A*x = b, the larger value
    ! btol*norm(b)/norm(r0)  should be suitable for A*dx = r0.
    !
    ! Preconditioning is another way to reduce the number of iterations.
    ! If it is possible to solve a related system M*x = b efficiently,
    ! where M approximates A in some helpful way
    ! (e.g. M - A has low rank or its elements are small relative to
    ! those of A), LSMR may converge more rapidly on the system
    !       A*M(inverse)*z = b,
    ! after which x can be recovered by solving M*x = z.
    ! Observe that, if options%ctest = 3 is used (Fong and Saunders stopping)
    ! then the convergence will be in the preconditioner norm.
    ! If options%ctest /= 3, the user may choose the norm that is
    ! used for the stopping criteria.
    !
    ! Currently, the user must combine applying the preconditioner
    ! with performing matrix-vector products
    ! eg suppose we have an incomplete factorization LL^T for A'A and we
    ! are going to use this as a preconditioner.
    ! Let the LS problem be  min||Ay - b||.
    ! When LSMR requires products A*v, the user should compute AL^{-T}*v.
    ! And when A'*v is required, the user should compute L^{-1}A'*v.
    ! On completion, the user must recover the required solution y
    ! by solving y = L^T*x, where x is the solution returned by LSMR.
    ! In a future release, we may include separate returns for preconditioning
    ! operations by extending the range of action values.
    ! This will allow LSMR to return y to the user.
    !
    ! NOTE: If A is symmetric, LSMR should not be used!
    ! Alternatives are the symmetric conjugate-gradient method (CG)
    ! and/or SYMMLQ.
    ! SYMMLQ is an implementation of symmetric CG that applies to
    ! any symmetric A and will converge more rapidly than LSMR.
    ! If A is positive definite, there are other implementations of
    ! symmetric CG that require slightly less work per iteration
    ! than SYMMLQ (but will take the same number of iterations).
    !
    !
    ! Notation
    ! --------
    ! The following quantities are used in discussing the subroutine
    ! parameters:
    !
    ! Abar   =  (  A   ),        bbar  =  (b)
    !           (damp*I)                  (0)
    !
    ! r      =  b - A*x,         rbar  =  bbar - Abar*x
    !
    ! normr  =  sqrt( norm(r)**2  +  damp**2 * norm(x)**2 )
    !        =  norm( rbar )
    !
    ! eps    =  the relative precision of floating-point arithmetic.
    !           On most machines, eps is about 1.0e-7 and 1.0e-16
    !           in single and double precision respectively.
    !           We expect eps to be about 1e-16 always.
    !
    ! LSMR  minimizes the function normr with respect to x.

! ------------------------------------------------------------------

! control data type
! These should be set before first call and must not be altered between calls.
  type, public :: lsmr_options_type

    integer(ip) :: ctest = 1
    ! Used to control convergence test. Possible values:
    ! 1 : User may test for convergence when action = 1 is returned.
    !     The code does NOT compute normA, condA, normr, normAr, normx.
    !     The code will only terminate if allocation (or deallocation) error
    !     or if itnlim is reached. Thus user is responsible for
    !     for determining when to stop.
    ! 2 : As 1 but inform holds the latest estimates of normA, condA, normr,  
    !     normAr, normx  which the user may wish to use to monitor convergence.
    ! 3 : The code determines if convergence has been achieved using 
    !     Fong and Saunders stopping criteria.
    !     inform holds the latest estimates of normA, condA, normr, normAr, 
    !     normx

    real(dp)    :: atol = sqrt(epsilon(one))  
    ! Used if ctest = 3 (Fong and Saunders stopping criteria)
    ! In which case, must hold an estimate of the relative error in the data
    ! defining the matrix A.  For example, if A is accurate to about 6 digits,
    ! set atol = 1.0e-6.

    real(dp)    :: btol = sqrt(epsilon(one)) 
    ! Used if ctest = 3 (Fong and Saunders stopping criteria)
    ! In which case, must hold an estimate of the relative error in the data
    ! defining the rhs b.  For example, if b is
    ! accurate to about 6 digits, set btol = 1.0e-6.

    real(dp)    :: conlim = 1/(10*sqrt(epsilon(one))) 
    ! Used if ctest = 3 (Fong and Saunders stopping criteria)
    ! In which case, must hold an upper limit on cond(Abar), the apparent
    ! condition number of the matrix Abar. Iterations will be terminated 
    ! if a computed estimate of cond(Abar) exceeds conlim.
    ! This is intended to prevent certain small or
    ! zero singular values of A or Abar from
    ! coming into effect and causing unwanted growth in the computed solution.
    !
    ! conlim and damp may be used separately or
    ! together to regularize ill-conditioned systems.
    !
    ! Normally, conlim should be in the range 1000 to 1/eps.
    ! Suggested value:
    ! conlim = 1/(100*eps)  for compatible systems,
    ! conlim = 1/(10*sqrt(eps)) for least squares.
    !
    ! Note: Any or all of atol, btol, conlim may be set to zero.
    ! The effect will be the same as the values eps, eps, 1/eps.
    !
    integer(ip) :: itnlim = 100000
    ! must hold an upper limit on the number of iterations.
    ! Suggested value:
    ! itnlim = n/2  for well-conditioned systems with clustered singular values
    ! itnlim = 4*n  otherwise.
    !
    integer(ip) :: localSize = 0 
    ! No. of vectors for local reorthogonalization.
    ! 0       No reorthogonalization is performed.
    ! >0      This many n-vectors "v" (the most recent ones)
    !         are saved for reorthogonalizing the next v.
    !         localSize need not be more than min(m,n).
    !         At most min(m,n) vectors will be allocated.
    !
    integer(ip) :: nout = 6 
    ! Unit number for printed output.  If positive,
    ! a summary will be printed on unit nout.

    integer(ip) :: pfreq  = 20 ! print frequency (for repeating the heading)
    !

  end type lsmr_options_type

! ------------------------------------------------------------------

! information data type
  type, public :: lsmr_inform_type

    integer(ip) :: istop ! Gives the reason for termination:
    !
    !            0       x = 0  is the exact solution.
    !                    No iterations were performed.
    !
    !            1       The equations A*x = b are probably compatible.
    !                    Norm(A*x - b) is sufficiently small, given the
    !                    values of atol and btol. ctest = 3 only.
    !
    !            2       If damp is zero:  The system A*x = b is probably
    !                    not compatible.  A least-squares solution has
    !                    been obtained that is sufficiently accurate,
    !                    given the value of atol.  
    !                    If damp is nonzero:  A damped least-squares
    !                    solution has been obtained that is sufficiently
    !                    accurate, given the value of atol. ctest = 3 only.
    !
    !            3       An estimate of cond(Abar) has exceeded conlim.
    !                    The system A*x = b appears to be ill-conditioned,
    !                    or there could be an error in Aprod1 or Aprod2.
    !                    options%ctest = 3 only.
    !
    !            4       Ax -b is small enough for this machine.
    !                    options%ctest = 3 only. 
    !
    !            5       The least-squares solution is good enough for this
    !                    machine. options%ctest = 3 only.
    !
    !            6       The estimate of cond(Abar) seems to be too large 
    !                    for this machine. options%ctest = 3 only.
    !
    !            7       The iteration limit options%itnlim has been reached.
    !
    !            8       An array allocation failed.
    !
    !            9       An array deallocation failed after the solution
    !                    estimate has been found
    !
    !
    integer(ip) :: itn    ! The number of iterations performed

    integer(ip) :: stat   ! Fortran stat parameter

    real(dp)    :: normb  ! holds norm of rhs

    real(dp)    :: normA  ! Only holds information if options%ctest = 2 or 3.
    ! In this case, holds estimate of the Frobenius norm of Abar.
    ! This is the square-root of the sum of squares of the elements of Abar.
    ! If damp is small and the columns of A have all been scaled to have 
    ! length 1.0, normA should increase to roughly sqrt(n).
    ! A radically different value for normA may
    ! indicate an error in the user-supplied
    ! products with A or A'. A negative value
    ! indicates that no estimate is currently available.
    !
    real(dp)    :: condA  ! Only holds information if options%ctest = 2 or 3.
    ! In this case, holds estimate of cond(Abar), the condition
    ! number of Abar.  A very high value of condA
    ! may again indicate an error in the products 
    ! with A or A'. A negative value indicates
    ! that no estimate is currently available.
    !
    real(dp)    :: normr  ! Only holds information if options%ctest = 2 or 3.
    ! In this case, holds estimate of the final value of norm(rbar),
    ! the function being minimized (see notation above).  This will be 
    ! small if A*x = b has a solution. A negative value
    ! indicates that no estimate is currently available.
    !
    real(dp)    :: normAr  ! Only holds information if options%ctest = 2 or 3.
    ! In this case, holds estimate of the final value of
    ! norm( Abar'*rbar ), the norm of the residual for the normal equations.
    ! This should be small in all cases.  (normAr  will often be smaller 
    ! than the true value computed from the output vector x.) A negative value
    ! indicates that no estimate is currently available.
    !
    real(dp)    :: normx ! Only holds information if options%ctest = 2 or 3.
    ! In this case, holds estimate of  norm(x) for the final solution x.
    ! A negative value indicates that no estimate is currently available.

  end type lsmr_inform_type

! ------------------------------------------------------------------

!  define derived type to ensure local variables are saved safely
  type, public :: lsmr_keep_type
    private
    real(dp), allocatable :: h(:), hbar(:), localV(:,:)
    logical     :: damped, localOrtho, localVQueueFull, show
    integer(ip) :: istop
    integer(ip) :: localOrthoCount, localOrthoLimit, localPointer,            &
                   localVecs, pcount
    real(dp)    :: alpha, alphabar, beta, betad, betadd, cbar, ctol,          &
                   d, maxrbar, minrbar, normA2, rho, rhobar, rhodold, sbar,   &
                   tautildeold, thetatilde, zeta, zetabar, zetaold
    integer(ip) :: branch = 0
  end type lsmr_keep_type

contains

  !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

  subroutine LSMR_reverse  ( m, n, action, u, v, b, damp, x, &
         keep, options, inform)

    integer(ip), intent(in)  :: m  ! the number of rows in A.

    integer(ip), intent(in)  :: n  ! the number of columns in A.

    integer(ip), intent(inout) :: action ! This parameter controls
    ! the action. On initial entry, must be set to 0.
    ! On subsequent call, action should be unchanged by the user UNLESS
    ! the user wishes to terminate the computation. In this case, it
    ! should be set to 3 and one further call to LSMR_reverse
    ! then made (see below).
    ! On return, action determines what the user is required to do.
    ! Possible values and their consequences are:
    ! 
    ! 0. Computation has terminated.
    !    action = 0 is returned if either an error has been encountered
    !    or convergence has been achieved with options%ctest = 3.
    !    It is also returned after a call with action = 3 (see below).
    !
    ! 1. The caller must compute v = v + A'*u without altering u, and 
    ! then re-call the subroutine. The vectors u and v are avaialble 
    ! in the arrays u and v (see below). Only v should be altered.
    !
    ! 2. The caller must compute u = u + A*v  without altering v, and 
    ! then re-call the subroutine. The vectors u and v are avaialble 
    ! in the arrays u and v (see below). Only u should be altered.
    !
    ! FINAL CALL: If the user wishes to deallocate components of keep
    ! a call with action = 10 should be made. Note: that if a series of
    ! problems is being solved, a call with action=10 is only needed
    ! after the final problem has been solved.

    real(dp),    intent(inout) :: u(m) ! The vector used to communicate u 
    !  when further action is required (see action, above).

    real(dp),    intent(inout) :: v(n) ! The vector used to communicate v 
    !  when further action is required (see action, above).

    real(dp),    intent(in)  :: b(m) ! The rhs vector b.

    real(dp),    intent(in)  :: damp ! The damping parameter for problem 3 
    ! above. (damp should be 0.0 for problems 1 and 2.) If the system A*x = b 
    ! is incompatible, values of damp in the range 0 to sqrt(eps)*norm(A)
    ! will probably have a negligible effect. Larger values of damp will tend 
    ! to decrease the norm of x and reduce the number of 
    ! iterations required by LSMR.
    !
    ! The work per iteration and the storage needed
    ! by LSMR are the same for all values of damp.

    real(dp),    intent(inout) :: x(n) ! Returns the computed solution x.

    type ( lsmr_keep_type ), intent( inout) :: keep 
    ! A variable of type lsmr_keep_type that is used to
    ! preserve internal data between reverse-communication
    ! calls. The components are private.

    type ( lsmr_options_type ), intent( inout) :: options
    ! A variable of type lsmr_options_type that is used to control the options.
    ! It should not be altered by the user.

    type ( lsmr_inform_type ), intent( inout) :: inform
    ! A variable of type lsmr_info_type that is used to hold information.
    ! It should not be altered by the user.

    intrinsic   :: abs, dot_product, min, max, sqrt

    ! Local arrays and variables
    integer(ip) :: i, localOrthoCount, st
    logical     :: prnt
    real(dp)    :: alphahat, betaacute, betacheck, betahat, c, chat,          &
                   ctildeold, rhobarold, rhoold, rhotemp, rhotildeold,        &
                   rtol, s, shat, stildeold, t1, taud, test1, test2, test3,   &
                   thetabar, thetanew, thetatildeold

    real(dp) :: dnrm2

    ! Local constants

    character(len=*), parameter :: enter = ' Enter LSMR.  '
    character(len=*), parameter :: exitt = ' Exit  LSMR.  '
    character(len=*), parameter :: msg(0:9 ) =                    &
      (/ 'The exact solution is  x = 0                         ', &
         'Ax - b is small enough, given atol, btol             ', &
         'The least-squares solution is good enough, given atol', &
         'The estimate of cond(Abar) has exceeded conlim       ', &
         'Ax - b is small enough for this machine              ', &
         'The LS solution is good enough for this machine      ', &
         'Cond(Abar) seems to be too large for this machine    ', &
         'The iteration limit has been reached                 ', &
         'Allocation error                                     ', &
         'Deallocation error                                   ' /)
    !-------------------------------------------------------------------

    ! on first call, initialize keep%branch and keep%istop
    if (action.eq.0) then
      keep%branch = 0
      keep%istop = 0
    end if

    ! Immediate return if we have already had an error 
    if (keep%istop.gt.0) return

    ! on final call, jump to deallocate components of keep
    if (action.eq.10) go to 900

    ! on other calls, jump to the appropriate place after 
    ! reverse-communication

    select case ( keep%branch )
      case ( 1 )
        go to 10
      case ( 2 )
        go to 20
      case ( 3 )
        go to 30
    end select

    ! Initialize.
    inform%normb  = -one
    inform%normr  = -one
    inform%normx  = -one
    inform%normA  = -one
    inform%normAr = -one
    inform%condA  = -one
    inform%stat   = 0

    keep%localVecs = min(options%localSize,m,n)
    keep%show      = options%nout >= 0
    if (keep%show) then
       if (options%ctest.eq.3) then
          write(options%nout, 1000) enter,m,n,damp,options%atol,&
           options%btol,options%conlim,options%itnlim,keep%localVecs
       else
          write(options%nout, 1100) enter,m,n,damp,options%itnlim,keep%localVecs
       end if
    end if
    
    keep%pcount = 0            ! print counter
    keep%damped = damp > zero  !

    !  allocate workspace (only do this if arrays not already allocated
    !  eg for an earlier problem)

    if (.not. allocated(keep%h)) then
       allocate( keep%h(n), stat = inform%stat )
       if ( inform%stat /= 0 ) then
          inform%istop = 8
          keep%istop   = 8
          if (keep%show) write(options%nout,'(a)') msg(8)
          action = 0
          return
       end if
    else if (size(keep%h).lt.n) then
       deallocate (keep%h, stat = inform%stat)
       if (inform%stat /= 0) then
          inform%istop = 9
          keep%istop   = 9
          if (keep%show) write(options%nout,'(a)') msg(9)
          action = 0
          return
       else
          allocate( keep%h(n), stat = inform%stat )
          if ( inform%stat /= 0 ) then
             inform%istop = 8
             keep%istop   = 8
             if (keep%show) write(options%nout,'(a)') msg(8)
             action = 0
             return
          end if
       end if
    end if

    if (.not. allocated(keep%hbar)) then
       allocate( keep%hbar(n), stat = inform%stat )
       if ( inform%stat /= 0 ) then
          inform%istop = 8
          keep%istop   = 8
          if (keep%show) write(options%nout,'(a)') msg(8)
          action = 0
          return
       end if
    else if (size(keep%hbar).lt.n) then
       deallocate (keep%hbar, stat = inform%stat)
       if (inform%stat /= 0) then
          inform%istop = 9
          keep%istop   = 9
          if (keep%show) write(options%nout,'(a)') msg(9)
          action = 0
          return
       else
          allocate( keep%hbar(n), stat = inform%stat )
          if ( inform%stat /= 0 ) then
             inform%istop = 8
             keep%istop   = 8
             if (keep%show) write(options%nout,'(a)') msg(8)
             action = 0
             return
          end if
       end if
    end if

    if (keep%localVecs > 0) then
     if (.not. allocated(keep%localV)) then
       allocate( keep%localV(n,keep%localVecs), stat = inform%stat )
       if ( inform%stat /= 0 ) then
          inform%istop = 8
          keep%istop   = 8
          if (keep%show) write(options%nout,'(a)') msg(8)
          action = 0
          return
       end if
     else if (size(keep%localV,1).lt.n .or. &
       size(keep%localV,2).lt.keep%localVecs) then
       deallocate (keep%localV, stat = inform%stat)
       if (inform%stat /= 0) then
          inform%istop = 9
          keep%istop   = 9
          if (keep%show) write(options%nout,'(a)') msg(9)
          action = 0
          return
       else
          allocate( keep%localV(n,keep%localVecs), stat = inform%stat )
          if ( inform%stat /= 0 ) then
             inform%istop = 8
             keep%istop   = 8
             if (keep%show) write(options%nout,'(a)') msg(8)
             action = 0
             return
          end if
       end if
     end if
    end if

15 continue

    !-------------------------------------------------------------------
    ! Set up the first vectors u and v for the bidiagonalization.
    ! These satisfy  beta*u = b,  alpha*v = A(transpose)*u.
    !-------------------------------------------------------------------
    u(1:m) = b(1:m)
    v(1:n) = zero
    x(1:n) = zero

    keep%alpha  = zero
    keep%beta   = dnrm2 (m, u, 1)

    if (keep%beta > zero) then
       call dscal (m, (one/keep%beta), u, 1)
       action = 1                  ! call Aprod2(m, n, v, u), i.e., v = A'*u
       keep%branch = 1
       return

    else
      ! Exit if b=0.
       inform%normA = - one
       inform%condA = - one
       if (keep%show) write(options%nout,'(a)') msg(1)
       go to 800
    end if

 10 continue
    if (keep%beta > zero) then
       keep%alpha = dnrm2 (n, v, 1)
    end if

    if (keep%alpha > zero) then
       call dscal (n, (one/keep%alpha), v, 1)
!      keep%w = v
    end if

    ! Exit if A'b = 0.

    inform%istop  = 0
    keep%istop    = 0
    inform%itn    = 0

    if (options%ctest.eq.2 .or. options%ctest.eq.3) then
       inform%normr  = keep%beta
       inform%normAr = keep%alpha * keep%beta
       if (inform%normAr == zero) then
          inform%normx = zero
          if (keep%show) write(options%nout,'(a)') msg(1)
          go to 800
       end if
    else
       ! if options%ctest = 1, just set the inform parameters to zero
       if (keep%alpha * keep%beta == zero) then
          if (keep%show) write(options%nout,'(a)') msg(1)
          go to 800
       end if
    end if

    ! Initialization for local reorthogonalization.

    keep%localOrtho = .false.
    if (keep%localVecs > 0) then
       keep%localPointer    = 1
       keep%localOrtho      = .true.
       keep%localVQueueFull = .false.
       keep%localV(:,1)     = v
    end if

    ! Initialize variables for 1st iteration.

    keep%zetabar  = keep%alpha*keep%beta
    keep%alphabar = keep%alpha
    keep%rho      = 1
    keep%rhobar   = 1
    keep%cbar     = 1
    keep%sbar     = 0

    keep%h         = v
    keep%hbar(1:n) = zero

    ! Initialize variables for estimation of ||r||.

    keep%betadd      = keep%beta
    keep%betad       = 0
    keep%rhodold     = 1
    keep%tautildeold = 0
    keep%thetatilde  = 0
    keep%zeta        = 0
    keep%d           = 0

    ! Initialize variables for estimation of ||A|| and cond(A).

    keep%normA2  = keep%alpha**2
    keep%maxrbar = zero
    keep%minrbar = huge(one)

    inform%normb  = keep%beta

    ! Items for use in stopping rules (needed for control%options = 3 only).
    keep%ctol   = zero
    if (options%conlim > zero) keep%ctol = one/options%conlim

    ! Heading for iteration log.

    if (keep%show) then
       if (options%ctest.eq.3) then
          if (keep%damped) then
             write(options%nout,1300)
          else
             write(options%nout,1200)
          end if
          test1 = one
          test2 = keep%alpha/keep%beta
          write(options%nout,1500) &
             inform%itn,x(1),inform%normr,inform%normAr,test1,test2
       else if (options%ctest.eq.2) then
          if (keep%damped) then
             write(options%nout,1350)
          else
             write(options%nout,1250)
          end if
          write(options%nout,1500) inform%itn,x(1),inform%normr,inform%normAr
       else
          ! simple printing
          write(options%nout,1400)
          write(options%nout,1600) inform%itn,x(1)
       end if
    end if

    !===================================================================
    ! Main iteration loop.
    !===================================================================
100 continue

       inform%itn = inform%itn + 1

	  
       !----------------------------------------------------------------
       ! Perform the next step of the bidiagonalization to obtain the
       ! next beta, u, alpha, v.  These satisfy
       !     beta*u = A*v  - alpha*u,
       !    alpha*v = A'*u -  beta*v.
       !----------------------------------------------------------------
       call dscal (m,(- keep%alpha), u, 1)
       action = 2             !  call Aprod1(m, n, v, u), i.e., u = u + A*v
       keep%branch = 2
       return

 20    continue
       keep%beta   = dnrm2 (m, u, 1)

       if (keep%beta > zero) then
          call dscal (m, (one/keep%beta), u, 1)
          if (keep%localOrtho) then ! Store v into the circular buffer localV
             call localVEnqueue     ! Store old v for local reorthog'n of new v.
          end if
          call dscal (n, (- keep%beta), v, 1)
          action = 1               ! call Aprod2(m, n, v, u), i.e., v = v + A'*u
          keep%branch = 3
          return
       end if

 30    continue
       if (keep%beta > zero) then
          if (keep%localOrtho) then ! Perform local reorthogonalization of V.
             call localVOrtho       ! Local-reorthogonalization of new v.
          end if
          keep%alpha  = dnrm2 (n, v, 1)
          if (keep%alpha > zero) then
             call dscal (n, (one/keep%alpha), v, 1)
          end if
       end if
    
       ! At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.
    
       !----------------------------------------------------------------
       ! Construct rotation Qhat_{k,2k+1}.

       alphahat = d2norm(keep%alphabar, damp)
       chat     = keep%alphabar/alphahat
       shat     = damp/alphahat

       ! Use a plane rotation (Q_i) to turn B_i to R_i.

       rhoold   = keep%rho
       keep%rho = d2norm(alphahat, keep%beta)
       c        = alphahat/keep%rho
       s        = keep%beta/keep%rho
       thetanew = s*keep%alpha
       keep%alphabar = c*keep%alpha

       ! Use a plane rotation (Qbar_i) to turn R_i^T into R_i^bar.
    
       rhobarold = keep%rhobar
       keep%zetaold   = keep%zeta
       thetabar  = keep%sbar*keep%rho
       rhotemp   = keep%cbar*keep%rho
       keep%rhobar    = d2norm(keep%cbar*keep%rho, thetanew)
       keep%cbar      = keep%cbar*keep%rho/keep%rhobar
       keep%sbar      = thetanew/keep%rhobar
       keep%zeta      =   keep%cbar*keep%zetabar
       keep%zetabar   = - keep%sbar*keep%zetabar

       ! Update h, h_hat, x.

       keep%hbar  = keep%h - (thetabar*keep%rho/(rhoold*rhobarold))*keep%hbar
       x          = x + (keep%zeta/(keep%rho*keep%rhobar))*keep%hbar
       keep%h     = v - (thetanew/keep%rho)*keep%h

       ! Estimate ||r||.
    
       ! Apply rotation Qhat_{k,2k+1}.
       betaacute =   chat* keep%betadd
       betacheck = - shat* keep%betadd

       ! Apply rotation Q_{k,k+1}.
       betahat   =   c*betaacute
       keep%betadd    = - s*betaacute

       ! Apply rotation Qtilde_{k-1}.
       ! betad = betad_{k-1} here.

       thetatildeold = keep%thetatilde
       rhotildeold   = d2norm(keep%rhodold, thetabar)
       ctildeold     = keep%rhodold/rhotildeold
       stildeold     = thetabar/rhotildeold
       keep%thetatilde    = stildeold* keep%rhobar
       keep%rhodold       =   ctildeold* keep%rhobar
       keep%betad         = - stildeold*keep%betad + ctildeold*betahat

       ! betad   = betad_k here.
       ! rhodold = rhod_k  here.

       keep%tautildeold                                                       &
                = (keep%zetaold - thetatildeold*keep%tautildeold)/rhotildeold
       taud     = (keep%zeta - keep%thetatilde*keep%tautildeold)/keep%rhodold
       keep%d   = keep%d + betacheck**2

       if (options%ctest.eq.2 .or. options%ctest.eq.3) then
          inform%normr  = sqrt(keep%d + (keep%betad - taud)**2 + keep%betadd**2)
    
          ! Estimate ||A||.
          keep%normA2   = keep%normA2 + keep%beta**2
          inform%normA  = sqrt(keep%normA2)
          keep%normA2   = keep%normA2 + keep%alpha**2
    
          ! Estimate cond(A).
          keep%maxrbar    = max(keep%maxrbar,rhobarold)
          if (inform%itn > 1) then 
             keep%minrbar = min(keep%minrbar,rhobarold)
          end if
          inform%condA    = max(keep%maxrbar,rhotemp)/min(keep%minrbar,rhotemp)

          ! Compute norms for convergence testing.
          inform%normAr  = abs(keep%zetabar)
          inform%normx   = dnrm2(n, x, 1)
       end if


       if (inform%itn     >= options%itnlim) inform%istop = 7
       if (options%ctest.eq.3) then

          !----------------------------------------------------------------
          ! Test for convergence.
          !----------------------------------------------------------------

          ! Now use these norms to estimate certain other quantities,
          ! some of which will be small near a solution.

          test1   = inform%normr /inform%normb
          test2   = inform%normAr/(inform%normA*inform%normr)
          test3   = one/inform%condA

          t1      = test1/(one + inform%normA*inform%normx/inform%normb)
          rtol    = options%btol + &
                      options%atol*inform%normA*inform%normx/inform%normb

          ! The following tests guard against extremely small values of
          ! atol, btol or ctol.  (The user may have set any or all of
          ! the parameters atol, btol, conlim  to 0.)
          ! The effect is equivalent to the normal tests using
          ! atol = eps,  btol = eps,  conlim = 1/eps.

          if (one+test3 <=  one) inform%istop = 6
          if (one+test2 <=  one) inform%istop = 5
          if (one+t1    <=  one) inform%istop = 4

          ! Allow for tolerances set by the user.

          if (  test3   <= keep%ctol   ) inform%istop = 3
          if (   test2  <= options%atol) inform%istop = 2
          if (  test1   <= rtol        ) inform%istop = 1

       end if

       !----------------------------------------------------------------
       ! See if it is time to print something.
       !----------------------------------------------------------------
       prnt = .false.
       if (keep%show) then
          if (n            <=                40) prnt = .true.
          if (inform%itn   <=                10) prnt = .true.
          if (inform%itn   >= options%itnlim-10) prnt = .true.
          if (mod(inform%itn,10)  ==          0) prnt = .true.
          if (options%ctest.eq.3) then
             if (test3 <=  onept*keep%ctol   ) prnt = .true.
             if (test2 <=  onept*options%atol) prnt = .true.
             if (test1 <=  onept*rtol        ) prnt = .true.
          end if
          if (inform%istop /=  0) prnt = .true.

          if (prnt) then        ! Print a line for this iteration
             if (keep%pcount >= options%pfreq) then  ! Print a heading first
                keep%pcount = 0
                if (options%ctest.eq.3) then
                   if (keep%damped) then
                      write(options%nout,1300)
                   else
                      write(options%nout,1200)
                   end if
                else if (options%ctest.eq.2) then
                   if (keep%damped) then
                      write(options%nout,1350)
                   else
                      write(options%nout,1250)
                   end if
                else
                   write(options%nout,1400)
                end if
             end if
             keep%pcount = keep%pcount + 1
             if (options%ctest.eq.3) then
                write(options%nout,1500) &
                  inform%itn,x(1),inform%normr,inform%normAr,          &
                  test1,test2,inform%normA,inform%condA
             else if (options%ctest.eq.2) then
                write(options%nout,1500) inform%itn,x(1),inform%normr, &
                  inform%normAr,inform%normA,inform%condA
             else
                write(options%nout,1600) inform%itn,x(1)
             end if
          end if
       end if

       if (inform%istop == 0) go to 100

    !===================================================================
    ! End of iteration loop.
    !===================================================================

    ! Come here if inform%normAr = 0, or if normal exit, or iteration
    ! count exceeded.

800 continue
    keep%istop = inform%istop

    if (keep%show) then ! Print the stopping condition.
       if (options%ctest.eq.2 .or. options%ctest.eq.3) then 
          write(options%nout, 2000)                      &
               exitt,inform%istop,inform%itn,            &
               exitt,inform%normA,inform%condA,          &
               exitt,inform%normb, inform%normx,         &
               exitt,inform%normr,inform%normAr
       else
          write(options%nout, 2100)                      &
               exitt,inform%istop,inform%itn
       end if
       write(options%nout, 3000) exitt, msg(inform%istop)
    end if

    ! terminate
    action = 0
    return

!!!! Final call (action =3) justs deallocates components of keep

900 continue
    if (allocated(keep%hbar)) then
       deallocate( keep%hbar, stat = st )
       if ( st /= 0 ) then
          inform%stat  = st
          inform%istop = 9
          if (keep%show) write(options%nout,'(a)') msg(9)
       end if
    end if

    if (allocated(keep%h)) then
       deallocate( keep%h, stat = st )
       if ( st /= 0 ) then
          inform%stat  = st
          inform%istop = 9
          if (keep%show) write(options%nout,'(a)') msg(9)
       end if
    end if

    if (allocated(keep%localV)) then
       deallocate( keep%localV, stat = st )
       if ( st /= 0 ) then
          inform%stat  = st
          inform%istop = 9
          if (keep%show) write(options%nout,'(a)') msg(9)
       end if
    end if

    ! terminate
    action = 0
    keep%istop   = inform%istop
    return

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

 1000 format(// a, '     Least-squares solution of  Ax = b'       &
     / ' The matrix  A  has', i7, ' rows   and', i7, ' columns'   &
     / ' damp   =', es22.14         &
     / ' options%atol   =', es10.2, &
     / ' options%btol   =', es10.2, &
     / ' options%conlim =', es10.2  &
     / ' options%itnlim =', i10     &
     / ' options%localSize (no. of vectors for local reorthogonalization) =',i7)
 1100 format(// a, '     Least-squares solution of  Ax = b'       &
     / ' The matrix  A  has', i7, ' rows   and', i7, ' columns'   &
     / ' damp   =', es22.14         &
     / ' options%itnlim =', i10     &
     / ' options%localSize (no. of vectors for local reorthogonalization) =',i7)
 1200 format(/ "   Itn       x(1)            norm r         A'r   ", &
      ' Compatible    LS      norm A    cond A')
 1250 format(/ "   Itn       x(1)            norm r         A'r   ", &
      ' norm A    cond A')
 1300 format(/ "   Itn       x(1)           norm rbar    Abar'rbar", &
      ' Compatible    LS    norm Abar cond Abar')
 1350 format(/ "   Itn       x(1)           norm rbar    Abar'rbar", &
      ' norm Abar cond Abar')
 1400 format(/ "   Itn       x(1)")
 1500 format(i6, 2es17.9, 5es10.2)
 1600 format(i6, es17.9)
 2000 format(/ a, 5x, 'istop  =', i2,   15x, 'itn    =', i8      &
      /      a, 5x, 'normA  =', es12.5, 5x, 'condA  =', es12.5   &
      /      a, 5x, 'normb  =', es12.5, 5x, 'normx  =', es12.5   &
      /      a, 5x, 'normr  =', es12.5, 5x, 'normAr =', es12.5)
 2100 format(/ a, 5x, 'istop  =', i2,   15x, 'itn    =', i8)
 3000 format(a, 5x, a)

  contains

    function d2norm( a, b )

      real(dp)             :: d2norm
      real(dp), intent(in) :: a, b

      !-------------------------------------------------------------------
      ! d2norm returns sqrt( a**2 + b**2 )
      ! with precautions to avoid overflow.
      !
      ! 21 Mar 1990: First version.
      ! 17 Sep 2007: Fortran 90 version.
      ! 24 Oct 2007: User real(dp) instead of compiler option -r8.
      !-------------------------------------------------------------------

      intrinsic            :: abs, sqrt
      real(dp)             :: scale
      real(dp), parameter  :: zero = 0.0_dp

      scale = abs(a) + abs(b)
      if (scale == zero) then
         d2norm = zero
      else
         d2norm = scale*sqrt((a/scale)**2 + (b/scale)**2)
      end if

    end function d2norm

    !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    subroutine localVEnqueue

      ! Store v into the circular buffer keep%localV.

      if (keep%localPointer < keep%localVecs) then
         keep%localPointer = keep%localPointer + 1
      else
         keep%localPointer = 1
         keep%localVQueueFull = .true.
      end if
      keep%localV(:,keep%localPointer) = v

    end subroutine localVEnqueue

    !+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    subroutine localVOrtho

      ! Perform local reorthogonalization of current v.

      real(dp)  :: d

      if (keep%localVQueueFull) then
         keep%localOrthoLimit = keep%localVecs
      else
         keep%localOrthoLimit = keep%localPointer
      end if

      do localOrthoCount = 1, keep%localOrthoLimit
         d = dot_product(v,keep%localV(:,localOrthoCount))
         v = v    -    d * keep%localV(:,localOrthoCount)
      end do

    end subroutine localVOrtho

  end subroutine LSMR_reverse

end module LsmrReverse
