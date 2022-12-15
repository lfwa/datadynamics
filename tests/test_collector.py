from collector import collector_v0

from pettingzoo.test import api_test


def main():
    env = collector_v0.env()
    api_test(env, num_cycles=10, verbose_progress=True)


if __name__ == "__main__":
    main()
