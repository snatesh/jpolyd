module lsmr_c_shim

  use iso_c_binding, only : c_int, c_double

  use lsmrReverse, only : LSMR_reverse, &
       lsmr_keep_type, lsmr_options_type, lsmr_inform_type

  implicit none

  private

  public :: lsmr_c_set_options
  public :: lsmr_c_step

  ! Singleton reverse-communication state.
  ! This is intentionally simple and matches the initial ctypes/shared-library
  ! validation path. It is not thread-safe and supports one active solve per
  ! shared-library instance. The C++ wrapper below performs cleanup at the end
  ! of each solve before starting another one.
  type(lsmr_keep_type),    save :: keep
  type(lsmr_options_type), save :: options
  type(lsmr_inform_type),  save :: inform

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
