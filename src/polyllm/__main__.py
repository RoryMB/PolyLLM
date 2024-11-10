from . import polyllm

if __name__ == '__main__':
    print('Ollama: (Only lists locally pulled models)')
    for model in polyllm.ollama_models():
        print(' ', model)
    print()

    print('OpenAI: (Some listed models may not be chat models)')
    for model in polyllm.openai_models():
        print(' ', model)
    print()

    print('Google:')
    for model in polyllm.google_models():
        print(' ', model)
    print()

    print('Anthropic: (Last updated - November 9, 2024)')
    for model in polyllm.anthropic_models():
        print(' ', model)
