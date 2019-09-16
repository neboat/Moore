# -*- tab-width: 8 ; indent-tabs-mode: t -*-
MMTARGETS=\
 mm_java\
 mm_c_gcc_O0\
 mm_c_gcc_O1\
 mm_c_gcc_O2\
 mm_c_gcc_O3\
 mm_c_icc_O2\
 mm_c_icc_O3\
 mm_permute\
 mm_transpose\
 mm_sdac\
 mm_sdac_coarsen_transpose\
 mm_sdac_coarsen_transpose_vec\
 mm_tile\
 mm_ploops\
 mm_ploops_tile\
 mm_ploops_tile_transpose\
 mm_ptile_transpose\
 mm_dac\
 mm_dac_coarsen\
 mm_dac_coarsen_transpose\
 mm_dac_1k\
 mm_dac_coarsen_transpose_gccmavx\
 mm_outer\
 mm_mkl_blas \
 #mm_dac_coarsen_transpose_axAVX\
 #Blank line
RUNNABLE=mm_python $(MMTARGETS)

NOPE=mm_dac_coarsen_transpose_gccmavx
CPPFLAGS=-I../fastcode/include $(EXTRA_CPPFLAGS)

EXTRA_TARGETS= mm_ploops_1k mm_ploops_tile_1k mm_ploops_tile_transpose_1k

default: $(patsubst %,%.exe,$(MMTARGETS) $(EXTRA_TARGETS))
lint: $(patsubst %.c,%.lint,$(filter-out timedata.c, $(wildcard *.c)))
.phony: %.lint %.time %.mtime %.qtime
mm_outer_8_4_256_256.lint       \
mm_outer_muladd.lint            \
mm_sdac_coarsen_transpose.lint  \
 :                              LINTARGS+=--linelength=160

mm_c.lint                       \
mm_dac.lint                     \
mm_dac_coarsen.lint             \
mm_dac_coarsen_transpose.lint   \
mm_mkl_blas.lint                \
mm_outer.lint                   \
mm_permute.lint                 \
mm_ploops.lint                  \
mm_ploops_padded.lint           \
mm_ploops_tile.lint             \
mm_ploops_tile3.lint            \
mm_ploops_tile_localtranspose.lint \
mm_ploops_tile_localtranspose_permuteloop.lint \
mm_ploops_tile_padded.lint      \
mm_ploops_tile_transpose.lint   \
mm_ploops_tile_transpose_padded.lint \
mm_ptile_transpose.lint         \
mm_sdac.lint                    \
mm_sdac_coarsen.lint            \
mm_sdac_coarsen_transpose.lint  \
mm_tile.lint                    \
mm_tile_local.lint              \
mm_tile_transpose.lint          \
mm_transpose.lint               \
mm_vec.lint                     \
 :                              LINTARGS+=--filter=-runtime/threadsafe_fn

%.lint: %.c
	../cpplint/cpplint.py --linelength=100 $(LINTARGS) $<
mm_java.exe: mm_java.java
	gcj $< --main=$(patsubst %.java,%,$<) -o $@

# CC=icc
CC=$(GCC)
GCC=gcc
ICC=icc
# adding -xHost is better for icc, since it produces avx
CILK = -fcilkplus
CILKLIBS = # -lcilkrts -ldl
OPTFLAGS=-O3
CFLAGS=$(OPTFLAGS) -std=c99 -W -Wall -Werror

%.exe: %.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDFLAGS) $(LDLIBS)

$(MMTARGETS): | verify.h

# This variable is filled in a little at a time below, and at the end of the file is the rule that updates
# LDFLAGS and CFLAGS to make cilk work.
NEEDS_CILK=

# this is a little too clever:
#define needs_cilk_internal =
#$(1): CFLAGS+=$$(CILK)
#$(1): LDFLAGS+=$$(CILKLIBS)
#endef
#NEEDS_CILK = $(eval $(call needs_cilk_internal,$$(1)))
#$(call NEEDS_CILK,mm_tile)

mm_c_gcc_O1.exe: OPTFLAGS=-O1
mm_c_gcc_O1.exe: mm_c.c
	$(GCC) $(CFLAGS) $(CPPFLAGS) $< -o $@

mm_c_gcc_O2.exe: OPTFLAGS=-O2
mm_c_gcc_O2.exe: mm_c.c
	$(GCC) $(CFLAGS) $(CPPFLAGS) $< -o $@

#-mtune=native or even -mavx give better performance
mm_c_gcc_O3.exe: OPTFLAGS=-O3
mm_c_gcc_O3.exe: mm_c.c
	$(GCC) $(CFLAGS) $(CPPFLAGS) $< -o $@

mm_c_gcc_O0.exe: CFLAGS=-O0 -std=c99
mm_c_gcc_O0.exe: mm_c.c
	$(GCC) $(CFLAGS) $(CPPFLAGS) $< -o $@

mm_c_icc_O2.exe: CFLAGS=-O2 -std=c99
mm_c_icc_O2.exe: mm_c.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@

mm_c_icc_O3.exe: mm_c.c
	$(ICC) $(CFLAGS) $(CPPFLAGS) $< -o $@

mm_c_icc_O3.s: mm_c.c
	$(ICC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -S

mm_transpose: CFLAGS=-O3 -std=c99 
mm_transpose: mm_transpose.c
	$(GCC) $(CFLAGS) $(CPPFLAGS)  $< -o $@

mm_sdac_coarsen_transpose_vec.exe: OPTFLAGS+=-ffast-math

#NEEDS_CILK+=mm_tile
mm_tile:           OPTFLAGS+=-ffast-math
#NEEDS_CILK+=mm_tile_local
mm_tile_local:     OPTFLAGS+=-ffast-math
#NEEDS_CILK+=mm_tile_transpose
mm_tile_transpose: OPTFLAGS+=-ffast-math

NEEDS_CILK+=vectorizable_b

NEEDS_CILK+=mm_ploops

NEEDS_CILK+=mm_ploops_1k
mm_ploops_1k.exe: mm_ploops.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDLIBS) -Dn=1024 

NEEDS_CILK+=mm_ploops_tile
mm_ploops_tile.exe: OPTFLAGS+=-ffast-math

NEEDS_CILK+=mm_ploops_tile3_1024
mm_ploops_tile3_1024.exe: mm_ploops_tile3.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDLIBS) -Dtilesize=1024

NEEDS_CILK+=mm_ploops_tile3_512
mm_ploops_tile3_512: mm_ploops_tile3.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDLIBS) -Dtilesize=512

NEEDS_CILK+=mm_ploops_tile3_256
mm_ploops_tile3_256: mm_ploops_tile3.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDLIBS) -Dtilesize=256

NEEDS_CILK+=mm_ploops_tile3_64
mm_ploops_tile3_64: mm_ploops_tile3.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDLIBS) -Dtilesize=64

NEEDS_CILK+=mm_ploops_tile3_32
mm_ploops_tile3_32: mm_ploops_tile3.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDLIBS) -Dtilesize=32

NEEDS_CILK+=mm_ploops_tile3_16
mm_ploops_tile3_16: mm_ploops_tile3.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDLIBS) -Dtilesize=16

#mm_ploops_tile3_elided: mm_ploops_tile3.c
#	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -cilk-serialize

NEEDS_CILK+=mm_ploops_tile_1k
mm_ploops_tile_1k.exe: mm_ploops_tile.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ $(LDLIBS) -Dn=1024

NEEDS_CILK+=mm_ploops_tile_transpose mm_ploops_tile_transpose_1k

mm_ploops_tile_transpose_1k.exe: mm_ploops_tile_transpose.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -Dn=1024 $(LDLIBS)

NEEDS_CILK+=mm_ptile_transpose
mm_ptile_transpose: OPTFLAGS+=-ffast-math

NEEDS_CILK+=mm_dac
mm_dac: OPTFLAGS+=-ffast-math

NEEDS_CILK+=mm_dac_coarsen
mm_dac_coarsen.exe: OPTFLAGS+=-DNOTRANSPOSE
# mm_dac_coarsen.exe: OPTFLAGS+=-ffast-math
mm_dac_coarsen.exe: mm_dac_coarsen_transpose.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDLIBS) $< -o $@

NEEDS_CILK+=mm_dac_coarsen_transpose
# mm_dac_coarsen_transpose.exe: OPTFLAGS+=-ffast-math

NEEDS_CILK+=mm_dac_coarsen_transpose_vec
mm_dac_coarsen_transpose_vec.exe: OPTFLAGS+=-ffast-math
mm_dac_coarsen_transpose_vec.exe: mm_dac_coarsen_transpose.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDLIBS) $< -o $@

NEEDS_CILK+=mm_dac_1k
mm_dac_1k.exe: CPPFLAGS+=-Dn=1024
mm_dac_1k.exe: mm_dac.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDLIBS) $< -o $@ 

NEEDS_CILK+=mm_dac_coarsen_1k
mm_dac_coarsen_1k: CPPFLAGS+=-Dn=1024
mm_dac_coarsen_1k: mm_dac_coarsen.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDLIBS) $< -o $@

NEEDS_CILK+=mm_dac_coarsen_transpose_1k
mm_dac_coarsen_transpose_1k: CPPFLAGS+=-Dn=1024
mm_dac_coarsen_transpose_1k: mm_dac_coarsen_transpose.c
	$(CC) $(CPPFLAGS) $(CFLAGS) $(LDLIBS) $< -o $@

#mm_dac_coarsen_transpose_xhost: mm_dac_coarsen_transpose.c
#	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -xHost

mm_dac_coarsen_transpose_axAVX: mm_dac_coarsen_transpose.c
	$(ICC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -axAVX

NEEDS_CILK+=mm_dac_coarsen_transpose_gccmavx
mm_dac_coarsen_transpose_gccmavx.exe: OPTFLAGS+=-ffast-math -mavx
mm_dac_coarsen_transpose_gccmavx.exe: mm_dac_coarsen_transpose.c
	$(GCC) $(CFLAGS) $(CPPFLAGS) $(LDLIBS) $< -o $@

NEEDS_CILK+=mm_outer
mm_outer.exe: CFLAGS  += -Wno-unknown-pragmas
mm_outer.exe: OPTFLAGS+= -march=native
mm_outer.exe: mm_outer.c mm_outer_muladd.c mm_outer_addstore.c mm_outer_8_4_256_256.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@

NEEDS_CILK+=mm_outer_strassen
mm_outer_strassen.exe: CFLAGS  += -Wno-unknown-pragmas
mm_outer_strassen.exe: OPTFLAGS+= -march=native
mm_outer_strassen.exe: mm_outer_strassen.c mm_outer_muladd.c mm_outer_addstore.c mm_outer_8_4_256_256.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@

mm_mkl_blas.exe: CC=$(ICC)
mm_mkl_blas.exe: LDLIBS=-mkl
#mm_mkl_blas.exe: LDLIBS=-lmkl_core -lmkl_intel_thread -lmkl_intel_lp64 -liomp5 -lm

VERIFYTARGETS=$(patsubst %,%.verify,$(RUNNABLE))

# "make verify" checks to see if the fancy code produces the same answer (up to numerical precision) as the slow code.
check: verify
.PHONY: check
verify: $(VERIFYTARGETS) $(patsubst %,%.verify,$(EXTRA_TARGETS))
.PHONY: verify
mm_python.verify mm_java.verify y mm_c_gcc_O0.verify mm_c_gcc_O3.verify mm_c_icc_O3.verify:
	@echo no verification needed for $@ > /dev/null

# "make screen" runs cilkscreen on the cilk codes
screen: mm_ploops_1k.screen mm_ploops_tile_1k.screen mm_ploops_tile_transpose_1k.screen mm_dac_1k.screen mm_dac_coarsen_1k.screen mm_dac_coarsen_transpose_1k.screen
%.screen: %
	cilkscreen ./$<

%.verify: %.exe
	./$< --verify

reverse = $(if $(1),$(call reverse,$(wordlist 2,$(words $(1)),$(1)))) $(firstword $(1))
TIMETARGS = $(call reverse, mm_python $(MMTARGETS))
foo:
	echo $(TIMETARGS)
time: $(patsubst %,%.time,$(TIMETARGS))

SOURCENAME = $(if $(or $(filter mm_c_icc_O%,$(1)),  \
                       $(filter mm_c_gcc_O%,$(1))), \
                  mm_c.c,                           \
                  $(if $(filter mm_dac_coarsen%,$(1)), \
                       mm_dac_coarsen_transpose.c, \
                       $(if $(filter mm_dac_coarsen_transpose%,$(1)), \
                            mm_dac_coarsen_transpose.c, \
                            $(if $(filter mm_ploops_2P,$(1)), \
                                 mm_ploops.c, \
                                 $(patsubst %.exe,%.c,$(1))))))

QUIESCE = $(if $(or $(filter mm_c_icc_O%,$(1)),  \
                    $(filter mm_c_gcc_O%,$(1))), \
               taskset -c 9, CILK_NWORKERS=16 taskset -c 0-15)

foo2:
	echo $(call SOURCENAME, mm_c_gcc_O0) $(call SOURCENAME, mm_java)

TIMECOUNT=5

# The time rules rely on the fact that the only thing printed by the program is a decimal number.
%.time: %.exe
	for x in $$(seq 1 $(TIMECOUNT)); do echo "{\""$$(hostname)"\",\""$$(date)"\",\""$(patsubst %.exe,%,$<)"\",\""$$(md5sum $(call SOURCENAME, $<) |cut -c1-32)"\"," $$(./$<) "}," >> timedata.c; done
%.mtime: %.exe
	for x in $$(seq 1 $(TIMECOUNT)); do echo "{\""$$(hostname)"\",\""$$(date)"\",\""$(patsubst %.exe,%,$<)"\",\""$$(md5sum $(call SOURCENAME, $<) |cut -c1-32)"\"," $$(./$<) "}," >> timedata_moore.c; done
%.qtime: %.exe
	for x in $$(seq 1 $(TIMECOUNT)); do echo "{\""$$(hostname)"\",\""$$(date)"\",\""$(patsubst %.exe,%,$<)"\",\""$$(md5sum $(call SOURCENAME, $<) |cut -c1-32)"\"," $$($(call QUIESCE, $<) ./$<) "}," >> qtimedata.c; done
mm_python.time: mm_python.py
	for x in $$(seq 1 $(TIMECOUNT)); do echo "{\""$$(hostname)"\",\""$$(date)"\",\""mm_python"\",\""$$(md5sum $<   |cut -c1-32)"\"," $$(python ./$<) "}," >> timedata.c; done
mm_java.time: mm_java.exe
	for x in $$(seq 1 $(TIMECOUNT)); do echo "{\""$$(hostname)"\",\""$$(date)"\",\""$<"\",\""$$(md5sum $<   |cut -c1-32)"\"," $$(./$<) "}," >> timedata.c; done
mm_python.mtime: mm_python.py
	for x in $$(seq 1 1); do echo "{\""$$(hostname)"\",\""$$(date)"\",\""mm_python"\",\""$$(md5sum $<   |cut -c1-32)"\"," $$(python ./$<) "}," >> timedata_moore.c; done
mm_java.mtime: mm_java.exe
	for x in $$(seq 1 $(TIMECOUNT)); do echo "{\""$$(hostname)"\",\""$$(date)"\",\""$(patsubst %.exe,%.java,$<)"\",\""$$(md5sum $<   |cut -c1-32)"\"," $$(./$<) "}," >> timedata_moore.c; done

# Run with two processors
mm_ploops_2P.time: mm_ploops
	for x in 1 2 3 4 5; do echo "{\""$$(hostname)"\",\""$$(date)"\",\""$(patsubst %.time,%,$@)"\",\""$$(md5sum $(call SOURCENAME, $<) |cut -c1-32)"\"," $$(CILK_NWORKERS=2 ./$<) "}," >> timedata.c; done

mm_mkl_blas_shared.time: mm_mkl_blas mm_c
	for x in $$(seq 1 $(TIMECOUNT)); do echo "{\""$$(hostname)"\",\""$$(date)"\",\""$(patsubst %.time,%,$@)"\",\""$$(md5sum $(call SOURCENAME, $<) |cut -c1-32)"\"," $$(./mm_c > /dev/null & ./mm_mkl_blas ; kill %1) "}," >> timedata.c; done


mmperf%: CFLAGS=-W -Wall -Werror -std=c99

mmperf.tex: mmperf
	./$< > $@

mmperf%.tex: mmperf%
	./$< > $@

.PHONY: clean
clean:
	rm -f $(MMTARGETS) mm_java.class *.S mm_*1k *.pyg *.exe

$(patsubst %,%.exe,$(NEEDS_CILK)): CFLAGS+=$(CILK)
$(patsubst %,%.exe,$(NEEDS_CILK)): LDLIBS+=$(CILKLIBS)
