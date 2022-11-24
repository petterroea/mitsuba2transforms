# mitsuba2transforms

Small helper script that generates camera transform matrices from mitsuba scenes, compatible with `instant-ngp`

Useful when training NERF on mitsuba scenes.

Tested with Mitsuba 0.6

## Dependencies

 * numpy
 * scipy

## Usage

```sh
python mitsuba2transforms.py project.xml images/00001.png transforms.json
```

Or maybe in a script that renders an animation and generates camera poses
```sh
#!/bin/zsh

TEMP_DIR=$(mktemp -d)
echo "Tempdir: $TEMP_DIR"

PROJECT_TEMPLATE=$1
NERF_DIR=$2

# No STEAMROOT mistakes here, please!
rm -rf $NERF_DIR

mkdir -p $NERF_DIR/images

echo "Creating scene files...\n"

for frame in {1..40}
do
	YAW=$((($frame*29)%180))
	PITCH=$(((($frame*29)/180)*13))
	echo -n "\r$frame / 80 ($PITCH $YAW)"
	jinja -D yaw $YAW -D pitch $PITCH $PROJECT_TEMPLATE > $TEMP_DIR/frame_$frame.xml
done

for frame in {1..40}
do
	fname=$(printf "%05d" $frame)
	echo "Rendering $fname"
	mitsuba $TEMP_DIR/frame_$frame.xml -o $NERF_DIR/images/$fname
	mtsutil tonemap $NERF_DIR/images/$fname.exr
	# Generate transform
	python ~/git/uni/mitsuba2transforms/mitsuba2transforms.py $TEMP_DIR/frame_$frame.xml $NERF_DIR/images/$fname.png $NERF_DIR/transforms.json
done
```