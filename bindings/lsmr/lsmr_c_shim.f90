module lsmr_c_shim

  use iso_c_binding, only : c_int, c_double

  use lsmrReverse, only : LSMR_reverse, &
       lsmr_keep_type, lsmr_options_type, lsmr_inform_type

  implicit none

  private

  public :: lsmr_c_set_options
  public :: lsmr_c_step

  ! One reverse-communication state per calling thread.
  !
  ! The public C ABI is unchanged: every solve still calls
  ! lsmr_c_set_options followed by repeated lsmr_c_step calls.  OpenMP
  ! THREADPRIVATE gives each thread independent keep/options/inform objects,
  ! including independent allocatable components owned by keep.
  !
  ! Requirements:
  !   * Compile this translation unit with OpenMP enabled.
  !   * Link the final shared library/executable with the OpenMP runtime.
  !   * All calls belonging to one solve, including action=10 cleanup, must
  !     execute on the same thread.
  !   * There may be one active solve per thread.  The shim is thread-safe but
  !     is not recursively reentrant within one thread.
  !
  ! nout remains disabled below, avoiding concurrent writes through a shared
  ! Fortran output unit.
  type(lsmr_keep_type),    save :: keep
  type(lsmr_options_type), save :: options
  type(lsmr_inform_type),  save :: inform

  !$omp threadprivate(keep, options, inform)

contains

  subroutine lsmr_c_set_options(atol, btol, conlim, itnlim, nout, localsize, ctest) &
       bind(C, name="lsmr_c_set_options")

    real(c_double), intent(in) :: atol
    real(c_double), intent(in) :: btol
    real(c_double), intent(in) :: conlim

    integer(c_int), intent(in) :: itnlim
    integer(c_int), intent(in) :: nout
    integer(c_int), intent(in) :: localsize
    integer(c_int), intent(in) :: ctest

    options%atol      = atol
    options%btol      = btol
    options%conlim    = conlim
    options%itnlim    = itnlim
    !options%nout      = nout
    options%nout = -1_c_int
    options%localsize = localsize
    options%ctest     = ctest

  end subroutine lsmr_c_set_options


  subroutine lsmr_c_step(m, n, action, u, v, b, damp, x, &
       istop, itn, stat, normr, normA, condA, normb, normx, normAr) &
       bind(C, name="lsmr_c_step")

    integer(c_int), intent(in)    :: m
    integer(c_int), intent(in)    :: n
    integer(c_int), intent(inout) :: action

    real(c_double), intent(inout) :: u(*)
    real(c_double), intent(inout) :: v(*)
    real(c_double), intent(in)    :: b(*)
    real(c_double), intent(in)    :: damp
    real(c_double), intent(inout) :: x(*)

    integer(c_int), intent(out) :: istop
    integer(c_int), intent(out) :: itn
    integer(c_int), intent(out) :: stat

    real(c_double), intent(out) :: normr
    real(c_double), intent(out) :: normA
    real(c_double), intent(out) :: condA
    real(c_double), intent(out) :: normb
    real(c_double), intent(out) :: normx
    real(c_double), intent(out) :: normAr

    call LSMR_reverse(m, n, action, u, v, b, damp, x, &
         keep, options, inform)

    istop = inform%istop
    itn   = inform%itn
    stat  = inform%stat

    normr  = inform%normr
    normA  = inform%normA
    condA  = inform%condA
    normb  = inform%normb
    normx  = inform%normx
    normAr = inform%normAr

  end subroutine lsmr_c_step

end module lsmr_c_shim
