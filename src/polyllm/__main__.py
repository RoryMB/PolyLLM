from . import polyllm

if __name__ == '__main__':
    polyllm.lazy_load()

    print('OpenAI:')
    for model in polyllm.openai_models:
        print(' ', model)
    print()

    print('Google:')
    for model in polyllm.google_models:
        print(' ', model)
    print()

    print('Anthropic:')
    for model in polyllm.anthropic_models:
        print(' ', model)
