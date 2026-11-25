OS := $(shell uname -s)

ifeq ($(OS),Darwin)
	BREW := $(shell brew --prefix)
	CFLAGS := `sdl2-config --libs --cflags` -I$(BREW)/include -L$(BREW)/lib -lGLEW -framework OpenGL -lpthread
else
	CFLAGS := `sdl2-config --libs --cflags` -lGLEW -lGL -lm -lpthread
endif

build:
	gcc voxel.c dos/*.c $(CFLAGS) -o voxel

run:
	./voxel

clean:
	rm voxel
