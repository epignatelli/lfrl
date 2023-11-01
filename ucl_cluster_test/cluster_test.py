import jax


def main():
    while True:
        x = jax.random.normal(jax.random.PRNGKey(0), (1000, 1000))
    return x


if __name__ == '__main__':
    main()