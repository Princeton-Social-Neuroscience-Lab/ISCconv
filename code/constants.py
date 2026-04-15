"""Project-level constants."""

TR = 1.5
SEED = 42

RUNS = (1, 2, 3, 4, 5)
TRIALS = (1, 2, 3, 4)
NTRIALS = len(TRIALS)
NRUNS = len(RUNS)

RUN_TRS = 544
TRIAL_TRS = 134  # excluding the first 12s (8 TR) blank on first trial of a run
CONV_TRS = 120
TRIAL_SLICES = [slice(i, i + 120) for i in range(0, 1200, 120)]

# Within each run, these are the indices of each trial, excluding prompt and blanks
RUN_TRIAL_SLICE = {
    1: slice(14, 134),
    2: slice(148, 268),
    3: slice(282, 402),
    4: slice(416, 536),
}

# filename keys, useful for ArgumentParser args object when updating Paths
FNKEYS = ("conv", "run", "trial")

# PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@\\^_`{|}~"  # string.punctuation without [ ]
PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

EXCLUDED_CONVS = (
    101,  # acquisition error, missing TRs
    119,  # aborted scan
    168,  # excessive movement
    171,  # missing runs due to malfunction. aborted scan
)

# These conversations had interruptions midway through or had to be restarted
# `ls -1 sourcedata/CONV_scan/data/TimingsLog | cut -d_ -f2 | uniq -c | sort -nr``
# 103, 104, 108, 111, 116, 117, 121, 122, 127, 129, 138, 143, 154, 164, 167
INTERRUPTED_CONVS = (103, 104, 108, 111, 116, 117, 121, 122, 127, 129, 138, 143, 154, 164, 167)  # friends + strangers

CONVS_STRANGERS = (
    104,
    105,
    106,
    107,
    108,
    111,
    112,
    114,
    116,
    117,
    120,
    122,
    123,
    126,
    128,
    129,
    131,
    132,
    133,
    137,
    138,
    142,
    143,
    153,
    156,
    157,
    158,
    163,
    174,
)
CONVS_FRIENDS = (
    103,
    109,
    113,
    118,
    121,
    127,
    130,
    134,
    145,
    146,
    147,
    148,
    150,
    151,
    152,
    154,
    155,
    159,
    160,
    161,
    162,
    164,
    165,
    166,
    167,
    169,
    170,
    172,
    173,
    175,
)
CONVS = CONVS_STRANGERS + CONVS_FRIENDS

SUBS_STRANGERS = list(CONVS_STRANGERS) + [c - 100 for c in CONVS_STRANGERS]
SUBS_FRIENDS = list(CONVS_FRIENDS) + [c - 100 for c in CONVS_FRIENDS]

SUBS = SUBS_FRIENDS + SUBS_STRANGERS


