# Comanche Satelllite Live Map

Generate real-world terrain maps from Sentinel-2 L2A and Copernicus GLO-30 DEM satellites and render them in a Voxel Space Engine, inspired by Comanche.


https://github.com/user-attachments/assets/7154251f-7aa7-44c1-aeb6-c730c02527de



## Prerequisites

To run the software, the required C graphics libraries and `uv` need to be installed.

```bash
brew install glew sdl2-config uv
```

## Running the software

The voxel space engine needs a color and height map to render the terrain.
Generate these assets before compiling and running the program.

```bash
# 1. Generating the colormap and heightmap
make generate LOCATION="Mount Rainier, Washington" COLORMAP="assets/map1_color.gif" HEIGHTMAP="assets/map1_height.gif"

# 2. Compiling and running the engine
make compile
make run
```

## Controls

| Key           | Action        | Effect                                      |
| :---          | :---          | :---                                        |
| `Up Arrow`    | Move forward  | Advances camera along current heading       |
| `Down Arrow`  | Move backward | Moves camera backward along current heading |
| `Left Arrow`  | Turn left     | Decreases camera yaw angle                  |
| `Right Arrow` | Turn right    | Increases camera yaw angle                  |
| `E`           | Ascend        | Increases camera altitude                   |
| `D`           | Descend       | Decreases camera altitude                   |
| `S`           | Pitch up      | Shifts horizon line down                    |
| `W`           | Pitch down    | Shifts horizon line up                      |


