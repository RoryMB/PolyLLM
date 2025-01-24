from . import polyllm

if __name__ == '__main__':
    for name, provider in polyllm.providers.items():
        print(name)
        for model in provider.get_models():
            print(' ', model)
        print()
