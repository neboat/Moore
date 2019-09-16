OPTFLAGS=-O3
CFLAGS=$(OPTFLAGS) -std=c99 -W -Wall -Werror
CC=icc
CPPFLAGS=-I. -I..
default: mm_outer_prefetch.exe mm_outer_prefetch_8K.exe
mm_outer_prefetch.exe: ../mm_outer.c ../mm_outer_muladd.c ../mm_outer_addstore.c mm_outer_8_4_256_256.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -o $@ -lcilkrts -ldl

mm_outer_prefetch_8K.exe: ../mm_outer.c ../mm_outer_muladd.c ../mm_outer_addstore.c mm_outer_8_4_256_256.c
	$(CC) $(CFLAGS) $(CPPFLAGS) $< -Dn=8192 -o $@ -lcilkrts -ldl
