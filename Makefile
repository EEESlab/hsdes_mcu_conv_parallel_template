PULP_APP = convolution
PULP_APP_SRCS = conv_kernel.c convolution.c

PULP_CFLAGS += -O3 -g

# if uncommented, the next line disables HW loops
# PULP_CFLAGS += -mnohwloop
# if uncommented, the next line disables post-increments
# PULP_CFLAGS += -mnopostmod
# if uncommented, the next line disables the MAC instruction
# PULP_CFLAGS += -mnomac

include $(PULP_SDK_HOME)/install/rules/pulp.mk
