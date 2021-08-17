import yaml


class AttrDict(dict):
    """Attr dict: make value private
    """

    def __init__(self, d):
        self.dict = d

    def __getattr__(self, attr):
        value = self.dict[attr]
        if isinstance(value, dict):
            return AttrDict(value)
        else:
            return value

    def __str__(self):
        return str(self.dict)


def load_config(config_file):
    """Load config file"""
    with open(config_file) as f:
        if hasattr(yaml, 'FullLoader'):
            config = yaml.load(f, Loader=yaml.FullLoader)
        else:
            config = yaml.load(f)
    print(config)
    return AttrDict(config)


def skip_gram_gen_pairs(walk, half_win_size=2):
    src, dst = list(), list()

    l = len(walk)
    # rnd = np.random.randint(1,  half_win_size+1, dtype=np.int64, size=l)
    for i in range(l):
        real_win_size = half_win_size
        left = i - real_win_size
        if left < 0:
            left = 0
        right = i + real_win_size
        if right >= l:
            right = l - 1
        for j in range(left, right + 1):
            if walk[i] == walk[j]:
                continue
            src.append(walk[i])
            dst.append(walk[j])
    return src, dst


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='text classification')
    parser.add_argument("-c", "--config", type=str, default="./config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
