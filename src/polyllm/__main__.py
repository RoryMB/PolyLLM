from . import polyllm

if __name__ == '__main__':
    for name, provider in polyllm.providers.items():
        models = provider.get_models()
        if not models:
            continue
        print(name)
        for model in models:
            print(' ', model)
        print()
