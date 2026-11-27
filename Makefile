OS := $(shell uname -s)

ifeq ($(OS),Darwin)
	BREW := $(shell brew --prefix)
	CFLAGS := `sdl2-config --libs --cflags` -I$(BREW)/include -L$(BREW)/lib -lGLEW -framework OpenGL -lpthread
else
	CFLAGS := `sdl2-config --libs --cflags` -lGLEW -lGL -lm -lpthread
endif

LOCATION ?= Mount St. Helens, Washington
COLORMAP ?= assets/map0_color.gif
HEIGHTMAP ?= assets/map0_height.gif

.PHONY: all generate compile run clean

# all: generate run

generate:
	uv run --project tools tools/generate_map.py \
		"$(LOCATION)" \
		--colormap_path "$(COLORMAP)" \
		--heightmap_path "$(HEIGHTMAP)"

compile:
	gcc engine/voxel.c engine/dos/*.c $(CFLAGS) -o voxel

run:
	./voxel

clean:
	rm voxel
