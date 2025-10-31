from dataclasses import dataclass

@dataclass
class FreqBand():
    name: str
    fmin: float
    fmax: float

    def __getitem__(self, key) -> float:
        if key == 0:
            return self.fmin
        if key == 1:
            return self.fmax
        raise ValueError(f"key must be 0 or 1. Received {key}")

@dataclass
class FreqBands():
    bands: list[FreqBand]

    def __init__(self, freq_bands_dict: dict):
        self.bands = []
        for k, v in freq_bands_dict.items():
            self.bands.append(FreqBand(k, v[0], v[1]))

    @property
    def as_dict(self):
        out = {}
        for band in self.bands:
            out[band.name] = [band.fmin, band.fmax]
        return out
    
    def __getitem__(self, key) -> FreqBand:
        if isinstance(key, str):
            for band in self.bands:
                if band.name == key:
                    return band
            raise ValueError(f"Cannot find frequency band with key '{key}'")
        return self.bands[key]
    
    def __len__(self) -> int:
        return len(self.bands)
    
    def __iter__(self):
        return iter(self.bands)
