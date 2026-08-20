# psd-laruen

## Setup

You will need some cabinet IR's to get the full experience.               
You can get some for free from [ML Sound Lab](https://ml-sound-lab.com/pages/free-premium-ir)

```
./scripts/download_data.sh

./scripts/compile_dataset.sh
```

## Example usage
```
uv run play stream clean assets/cabinets/Tube\ Color.wav data/Ibanez2820-DI.wav

uv run play process metal assets/cabinets/Tube\ Color.wav data/Ibanez2820-DI.wav -o output.wav
```

