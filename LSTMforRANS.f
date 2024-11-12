!************************************************************************************************************
!
!     This software predicts the Reynolds stress \(\langle uv \rangle^+\) for developed and developing 2D turbulent
!      flows using sequences of 32 elements, processed through the LSTM model, as defined in LSTM_params_4XBL13BL23BL33DL_32.txt.

!     The feature inputs (\(xx(:, :)\)) and the corresponding predictions (\(yy(:)\)) represent the sequences.
!     Features should be normalized using the maximum and minimum values from the file maxmin_4XBL13BL23BL33DL.dat.
!     The normalization follows the formula:  
!     \[
!     \text{normalized\_val} = \frac{\text{val} - \text{min}}{\text{max} - \text{min}}.
!     \]
!     Since the predicted values stored in \(yy(:)\) are also normalized, they should be de-normalized using:  
!     \[
!     \text{val} = yy(:) \times (\text{max} - \text{min}) + \text{min}.
!     \]

!     The feature sequences are structured as:
!     \[
!     xx((Re_{\tau}, S_{11}^+, S_{12}^+, Y)_1, (Re_{\tau}, S_{11}^+, S_{12}^+, Y)_2, \dots, (Re_{\tau}, S_{11}^+, S_{12}^+, Y)_{32}),
!     \]
!     where \(Re_{\tau}\), \(S_{11}^+\), \(S_{12}^+\), and \(Y\) are the input variables at different time steps.
!     The corresponding predictions are stored as:  
!     \[
!     yy(uv^+_1, uv^+_2, \dots, uv^+_{32}).
!     \]

!************************************************************************************************************
      MODULE mod_kind
      IMPLICIT NONE
      INTEGER,PARAMETER::idp=KIND(1.d00)
      private
      public:: idp
      END MODULE mod_kind
!*********************************************
      MODULE mod_random
!     This modul was originally developed by M. Cursic
!     \bibitem [Curcic, 2016]{curcic2019} Curcic, M. 2019 A parallel Fortran
!     framwork for neural networks and deep learning,
!     modern-fortran/neural-fortran, arXiv:1902.06714; GitHub.
      
       USE mod_kind
      
! Provides a random number generator with
! normal distribution, centered on zero.
          
       IMPLICIT NONE
      
       private
       public :: randn
      
      real(idp), parameter :: pi = 4 * atan(1._idp)
      
      interface randn
      module procedure :: randn1d, randn2d
      end interface randn
      
      contains
      
      function randn1d(n) result(r)
! Generates n random numbers with a normal distribution.
      integer, intent(in) :: n
      real(idp):: r(n), r2(n)
      call random_number(r)
      call random_number(r2)
      r = sqrt(-2 * log(r)) * cos(2 * pi * r2)
      end function randn1d
      
      function randn2d(m, n) result(r)
! Generates m x n random numbers with a normal distribution.
      integer, intent(in) :: m, n
      real(idp) :: r(m, n), r2(m, n)
      call random_number(r)
      call random_number(r2)
      r = sqrt(-2 * log(r)) * cos(2 * pi * r2)
      end function randn2d

      END MODULE mod_random
!*****************************************************************************
      MODULE mod_function
!     This modul was originally developed by M. Cursic
!     \bibitem [Curcic, 2016]{curcic2019} Curcic, M. 2019 A parallel Fortran
!     framwork for neural networks and deep learning,
!     modern-fortran/neural-fortran, arXiv:1902.06714; GitHub.

      USE mod_kind
      
      IMPLICIT NONE
      
      contains
  
      PURE FUNCTION sigmoid(x) result(res)
! Sigmoid activation function.
      
      REAL(idp), intent(in) :: x
      REAL(idp) :: res
      res = 1 / (1 + exp(-x))
      ENDFUNCTION sigmoid
!------------------------------------------------
      PURE FUNCTION sigmoid_prime(x) result(res)
! First derivative of the sigmoid activation function.
      
      real(idp), intent(in) :: x
      real(idp) :: res
      res = sigmoid(x) * (1 - sigmoid(x))
      end function sigmoid_prime
!-------------------------------------------
      pure function tanhf(x) result(res)
! Tangent hyperbolic activation function. 
      
      real(idp), intent(in) :: x
      real(idp) :: res
      res = tanh(x)
      end function tanhf
!------------------------------------------------
      pure function tanh_prime(x) result(res)
! First derivative of the tanh activation function.
      
      real(idp), intent(in) :: x
      real(idp) :: res
      res = 1 - tanh(x)**2
      end function tanh_prime
!-------------------------------------------
      pure function identity(x) result(res)
! Identity activation function. 
      
      real(idp), intent(in) :: x
      real(idp) :: res
      res = x
      end function identity
!------------------------------------------------
      pure function identity_prime(x) result(res)
! First derivative of the identity activation function.
      
      real(idp), intent(in) :: x
      real(idp) :: res
      res = 1 
      end function identity_prime
!-------------------------------------------
      pure function relu(x) result(res)
!! REctified Linear Unit (RELU) activation function.
      real(idp), intent(in) :: x
      real(idp) :: res
      res = max(0., x)
      end function relu
      
      pure function relu_prime(x) result(res)
! First derivative of the REctified Linear Unit (RELU) activation function.
      real(idp), intent(in) :: x
      real(idp) :: res
      IF (x > 0) THEN
         res = 1
      ELSE
         res = 0
      END IF
      end function relu_prime
!-------------------------------------------
      pure function lrelu(x) result(res)
!! REctified Linear Unit (RELU) activation function.
      real(idp), intent(in) :: x
      real(idp) :: res
      res = max(0.1*x, x)
      end function lrelu
      
      pure function lrelu_prime(x) result(res)
! First derivative of the Rectified Linear Unit (RELU) activation function.
      real(idp), intent(in) :: x
      real(idp) :: res
      IF (x >= 0) THEN
         res = 1
      ELSE
         res = 0.01
      END IF
      end function lrelu_prime
!-----------------------------------
      END MODULE mod_function
!*********************************************
      MODULE mod_general
      USE mod_kind

      IMPLICIT NONE
      
      INTEGER,PARAMETER:: T=32, NLSTM=3, MNMU=4 !T:terms of sequence, NLSTM:stacked LSTM; MNMU: maximum memory units of the LSTM layers
                                                ! or input features 
      REAL(idp),DIMENSION(T):: y,y_p
      
      REAL(idp),DIMENSION(MNMU,NLSTM,  0:T):: c,h
      REAL(idp),DIMENSION(MNMU,NLSTM,    T):: ft,it,ot,gt
      REAL(idp),DIMENSION(MNMU*2,NLSTM+1,T):: x
    
      REAL(idp),DIMENSION(MNMU,NLSTM  ,T+1):: Bc,Bh
      REAL(idp),DIMENSION(MNMU,NLSTM  ,  T):: Bft,Bit,Bot,Bgt
      
      REAL(idp):: zf,zi,zo,zg,zdl
      INTEGER:: l,i,j,k,ib,im,mu,muu
      REAL(idp):: fp,ip,gp,op,hp,cp,y_pp
      
      TYPE:: param_x
      REAL(idp):: f
      REAL(idp):: i
      REAL(idp):: o
      REAL(idp):: c
      END TYPE param_x
      
      TYPE:: param_h
      REAL(idp):: f
      REAL(idp):: i
      REAL(idp):: o
      REAL(idp):: c
      END TYPE param_h

      TYPE:: bias
      REAL(idp):: f
      REAL(idp):: i
      REAL(idp):: o
      REAL(idp):: c
      END TYPE bias
  
      TYPE (param_x),DIMENSION(MNMU,2*MNMU,NLSTM):: Wx
      TYPE (param_h),DIMENSION(MNMU,2*MNMU,NLSTM):: Wh
      TYPE (bias),DIMENSION(MNMU,NLSTM)   :: b
      
      TYPE (param_x),DIMENSION(MNMU,2*MNMU,NLSTM):: BWx
      TYPE (param_h),DIMENSION(MNMU,2*MNMU,NLSTM):: BWh
      TYPE (bias),DIMENSION(MNMU,NLSTM)   :: Bb

      REAL(idp), ALLOCATABLE :: xx(:,:),yy(:)

      REAL(idp),DIMENSION(2*MNMU):: Wdl
      REAL(idp):: dbdl_batch,dbdl,sdbdl,bdl
  
      INTEGER,DIMENSION(0:NLSTM):: NMU

      INTEGER,DIMENSION(1:NLSTM):: BWRAPS

      INTEGER,DIMENSION(1:NLSTM+1):: NIN
!----------------------------
      END MODULE mod_general
!***********************************************
      MODULE mod_LSTM
      Use mod_kind
      USE mod_general
      USE mod_function
      USE mod_random, only: randn
      IMPLICIT NONE

      REAL(idp):: Re_max,Re_min,S11_max,S11_min,S12_max,
     &  S12_min,yplus_max,yplus_min,uv_max,uv_min

      CONTAINS
  !---------------------------------------
      SUBROUTINE input_param
!     inputs the parameters of the LSTM
      IMPLICIT NONE
      
      REAL(idp):: W(4*MNMU*2*MNMU),U(4*MNMU*2*MNMU),bb_param(2*MNMU*4)
!-------------------------------------------
!...  Input max and min values
!-----------------------------------------
      NMU(0)=4
      NMU(1)=3
      NMU(2)=3
      NMU(3)=3
      
      BWRAPS(1)=1
      BWRAPS(2)=1
      BWRAPS(3)=1
      
      NIN(1)=4                                             !number of input fwatures(inputs at entrance)
      NIN(2)=NMU(1)+NMU(1)*BWRAPS(1)                       ! inputs at first layer
      NIN(3)=NMU(2)+NMU(2)*BWRAPS(2)                       ! inputs at second layer
      NIN(4)=NMU(3)+NMU(3)*BWRAPS(3)                       ! inputs at thrid layer

      h(:,:,:)=0
      c(:,:,:)=0
      Bh(:,:,:)=0
      Bc(:,:,:)=0

      OPEN( unit=29,file='maxmin_4XBL13BL23BL33DL.dat',form='formatted')

      READ(29,*) Re_max,Re_min,S11_max,S11_min,S12_max,S12_min,          !maximum and minimum of input features
     &       yplus_max,yplus_min,uv_max,uv_min
      CLOSE(29)

      OPEN(1,file='params_4XBL13BL23BL33DL_T32.txt')                     !LSTM
!------------------------------
!... input W_x parameters
!------------------------------
      through_the_LSTM_layers: DO l=1,NLSTM

      READ(1,'(1a)')
      READ(1,'(1a)')
      READ(1,*) (W(mu),mu=1,(4*NMU(l)*NIN(l)))
      DO mu=1,NIN(l)
         DO muu=1,NMU(l)
            Wx(muu,mu,l)%i=W(0*NMU(l)+4*(mu-1)*NMU(l)+1+(muu-1))
         END DO
      END DO
      
      DO mu=1,NIN(l)
         DO muu=1,NMU(l)
            Wx(muu,mu,l)%f=W(1*NMU(l)+4*(mu-1)*NMU(l)+1+(muu-1))
         END DO
      END DO
      
      DO mu=1,NIN(l)
         DO muu=1,NMU(l)
            Wx(muu,mu,l)%c=W(2*NMU(l)+4*(mu-1)*NMU(l)+1+(muu-1))
         END DO
      END DO
      
      DO mu=1,NIN(l)
         DO muu=1,NMU(l)
            Wx(muu,mu,l)%o=W(3*NMU(l)+4*(mu-1)*NMU(l)+1+(muu-1))
         END DO
      END DO
!------------------------------
!... input W_h parameters
!... I/O en Python, una maravillosa porqueria!!!
!------------------------------
      READ(1,'(1a)')
      READ(1,'(1a)')
      READ(1,*) (U(mu),mu=1,4*NMU(l)**2)
      DO mu=1,NMU(l)
         DO muu=1,NMU(l)
            Wh(muu,mu,l)%i=U(0*NMU(l)+4*(mu-1)*NMU(l)+1+(muu-1))
         END DO
      END DO
      DO mu=1,NMU(l)
         DO muu=1,NMU(l)
            Wh(muu,mu,l)%f=U(1*NMU(l)+4*(mu-1)*NMU(l)+1+(muu-1))
         END DO
      END DO
      DO mu=1,NMU(l)
         DO muu=1,NMU(l)
            Wh(muu,mu,l)%c=U(2*NMU(l)+4*(mu-1)*NMU(l)+1+(muu-1))
         END DO
      END DO
      DO mu=1,NMU(l)
         DO muu=1,NMU(l)
            Wh(muu,mu,l)%o=U(3*NMU(l)+4*(mu-1)*NMU(l)+1+(muu-1))
         END DO
      END DO
      
      READ(1,'(1a)')
      READ(1,'(1a)')
      READ(1,*) (bb_param(mu),mu=1,4*NMU(l))
      DO mu=1,NMU(l)
         b(mu,l)%i=bb_param(0*NMU(l)+1+(mu-1))
         b(mu,l)%f=bb_param(1*NMU(l)+1+(mu-1))
         b(mu,l)%c=bb_param(2*NMU(l)+1+(mu-1))
         b(mu,l)%o=bb_param(3*NMU(l)+1+(mu-1))
      END DO
!-------------------------------------------
!...Input parameters for the backward direction
!-----------------------------------------------
      IF(BWRAPS(l).EQ.1) THEN
         READ(1,'(1a)')
         READ(1,'(1a)')
         READ(1,*) (W(mu),mu=1,(4*NMU(l)*NIN(l)))
         DO mu=1,NIN(l)
            DO muu=1,NMU(l)
               BWx(muu,mu,l)%i=W(0*NMU(l)+4*(mu-1)*NMU(l)+1+(muu-1))
            END DO
         END DO
         
         DO mu=1,NIN(l)
            DO muu=1,NMU(l)
               BWx(muu,mu,l)%f=W(1*NMU(l)+4*(mu-1)*NMU(l)+1+(muu-1))
            END DO
         END DO
         
         DO mu=1,NIN(l)
            DO muu=1,NMU(l)
               BWx(muu,mu,l)%c=W(2*NMU(l)+4*(mu-1)*NMU(l)+1+(muu-1))
            END DO
         END DO
         
         DO mu=1,NIN(l)
            DO muu=1,NMU(l)
               BWx(muu,mu,l)%o=W(3*NMU(l)+4*(mu-1)*NMU(l)+1+(muu-1))
            END DO
         END DO
!------------------------------
!... input W_h parameters for the backward direction
!------------------------------
         READ(1,'(1a)')
         READ(1,'(1a)')
         READ(1,*) (U(mu),mu=1,4*NMU(l)**2)
         DO mu=1,NMU(l)
            DO muu=1,NMU(l)
               BWh(muu,mu,l)%i=U(0*NMU(l)+4*(mu-1)*NMU(l)+1+(muu-1))
            END DO
         END DO
         DO mu=1,NMU(l)
            DO muu=1,NMU(l)
               BWh(muu,mu,l)%f=U(1*NMU(l)+4*(mu-1)*NMU(l)+1+(muu-1))
            END DO
         END DO
         DO mu=1,NMU(l)
            DO muu=1,NMU(l)
               BWh(muu,mu,l)%c=U(2*NMU(l)+4*(mu-1)*NMU(l)+1+(muu-1))
            END DO
         END DO
         DO mu=1,NMU(l)
            DO muu=1,NMU(l)
               BWh(muu,mu,l)%o=U(3*NMU(l)+4*(mu-1)*NMU(l)+1+(muu-1))
            END DO
         END DO
         
         READ(1,'(1a)')
         READ(1,'(1a)')
         READ(1,*) (bb_param(mu),mu=1,4*NMU(l))
         DO mu=1,NMU(l)
            Bb(mu,l)%i=bb_param(0*NMU(l)+1+(mu-1))
            Bb(mu,l)%f=bb_param(1*NMU(l)+1+(mu-1))
            Bb(mu,l)%c=bb_param(2*NMU(l)+1+(mu-1))
            Bb(mu,l)%o=bb_param(3*NMU(l)+1+(mu-1))
         END DO
         
      END IF

      END DO through_the_LSTM_layers
      
      READ(1,'(1a)')
      READ(1,'(1a)')
      READ(1,*) (Wdl(mu),mu=1,NIN(NLSTM+1))
      
      READ(1,'(1a)')
      READ(1,'(1a)')
      READ(1,*) bdl
      CLOSE(1)
!------------------------------------
      END SUBROUTINE input_param
!----------------------------
      SUBROUTINE LSTM_prediction(xx,yy)
!this uses the LSTM to predict 
      USE mod_kind
      IMPLICIT NONE
      
      REAL(idp):: xx(:,:) !xx(:,:) input features of the sequence (should be previously normalized) 
      REAL(idp):: yy(:)   !yy(:)   output predictions 

!----------------------
!...prediction
!----------------------
      c=0
      h=0
      Bc=0
      Bh=0
      
      x(1:NIN(1), 1, :)=xx(1:NIN(1),:)
       
       through_LSTM_layers: DO l=1,NLSTM
    
          !---------------------------------------------
          !...Forward pass of LSTM layer
          !---------------------------------------------
          through_time_for: DO i=1,T
             
             through_memory_units_for: DO mu=1,NMU(l)         
                zf=sum(Wx(mu,1:NIN(l),l)%f*x(1:NIN(l),l,i  )) +    
     &            sum(Wh(mu,1:NMU(l),l)%f*h(1:NMU(l),l,i-1)) + b(mu,l)%f
                ft(mu,l,i)=sigmoid(zf)
                
                zi=sum(Wx(mu,1:NIN(l),l)%i*x(1:NIN(l),l,i  )) +    
     &            sum(Wh(mu,1:NMU(l),l)%i*h(1:NMU(l),l,i-1)) + b(mu,l)%i
                it(mu,l,i)=sigmoid(zi)
                
                zo=sum(Wx(mu,1:NIN(l),l)%o*x(1:NIN(l),l,i  )) +    
     &            sum(Wh(mu,1:NMU(l),l)%o*h(1:NMU(l),l,i-1)) + b(mu,l)%o
                ot(mu,l,i)=sigmoid(zo)
                
                zg=sum(Wx(mu,1:NIN(l),l)%c*x(1:NIN(l),l,i  )) +    
     &            sum(Wh(mu,1:NMU(l),l)%c*h(1:NMU(l),l,i-1)) + b(mu,l)%c
                gt(mu,l,i)=tanhf(zg)
                
                c(mu,l,i)=ft(mu,l,i)*c(mu,l,i-1)+it(mu,l,i)*gt(mu,l,i)
                
                h(mu,l,i)=ot(mu,l,i)*tanhf(c(mu,l,i))
                
                x(mu,l+1,i)=h(mu,l,i)

             END DO through_memory_units_for
             
          END DO through_time_for

          !---------------------------------------------
          !...Backward pass of bidirectional layer
          !---------------------------------------------
          IF(BWRAPS(l).EQ.1) THEN
             
             through_time_back: DO ib=T,1,-1
                
                through_memory_units_back: DO mu=1,NMU(l)         
                  zf=sum(BWx(mu,1:NIN(l),l)%f*x (1:NIN(l),l,ib  )) +    
     &        sum(BWh(mu,1:NMU(l),l)%f*Bh(1:NMU(l),l,ib+1)) + Bb(mu,l)%f
                   Bft(mu,l,ib)=sigmoid(zf)
                   
                  zi=sum(BWx(mu,1:NIN(l),l)%i*x (1:NIN(l),l,ib  )) +   
     &        sum(BWh(mu,1:NMU(l),l)%i*Bh(1:NMU(l),l,ib+1)) + Bb(mu,l)%i
                   Bit(mu,l,ib)=sigmoid(zi)
                   
                 zo=sum(BWx(mu,1:NIN(l),l)%o*x (1:NIN(l),l,ib  )) +    
     &        sum(BWh(mu,1:NMU(l),l)%o*Bh(1:NMU(l),l,ib+1)) + Bb(mu,l)%o
                   Bot(mu,l,ib)=sigmoid(zo)
                   
                 zg=sum(BWx(mu,1:NIN(l),l)%c*x (1:NIN(l),l,ib  )) +    
     &        sum(BWh(mu,1:NMU(l),l)%c*Bh(1:NMU(l),l,ib+1)) + Bb(mu,l)%c
                   Bgt(mu,l,ib)=tanhf(zg)
                   
        Bc(mu,l,ib)=Bft(mu,l,ib)*Bc(mu,l,ib+1)+Bit(mu,l,ib)*Bgt(mu,l,ib)
                   
                   Bh(mu,l,ib)=Bot(mu,l,ib)*tanhf(Bc(mu,l,ib))
                   
                   x(NMU(l)+mu,l+1,ib)= Bh(mu,l,ib)       

             END DO through_memory_units_back
                         
             END DO through_time_back
             
          END IF
          
       END DO through_LSTM_layers
    
       DO i=1,T
          zdl = sum(Wdl(1:NIN(NLSTM+1))*x(1:NIN(NLSTM+1),NLSTM+1,i))+bdl
          y_p(i)=identity(zdl)
          yy(i)=y_p(i)                                                       !predictions of the sequence (they are normalized)
       END DO
!---------------------------
       END SUBROUTINE LSTM_prediction
!------------------------------
      END MODULE mod_LSTM
!************************************************************************
