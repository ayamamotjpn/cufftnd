!  test program for cufftnd called from fortran
!  using quasicrystal data
!  check FFT and invers FFT in 4D octagonal QCs.
!  Structure factor of all reflections
!  is read from the input file (so that symmetry
!   operation is necessary).
!  Necessary memory is estimated by similarity
!  transformation of Miller indices.
!  Calculate total electron density by cufftnd
!  and compare speed with reference fftw.
!


