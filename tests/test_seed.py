from collector import collector_v0

from pettingzoo.test import seed_test


def test_seed():
    env = collector_v0.env()
    seed_test(env, num_cycles=10, test_kept_state=True)


if __name__ == "__main__":
    test_seed()
