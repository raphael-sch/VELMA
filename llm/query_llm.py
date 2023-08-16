import json
import os
import time

import openai


class LLM:
    def __init__(self, api_key, model_name, max_tokens, cache_name='default', **kwargs):
        self.api_key = api_key
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.queried_tokens = 0

        cache_model_dir = os.path.join('llm', 'cache', self.model_name)
        os.makedirs(cache_model_dir, exist_ok=True)
        self.cache_file = os.path.join(cache_model_dir, f'{cache_name}.json')
        self.cache = dict()

        if os.path.isfile(self.cache_file):
            with open(self.cache_file) as f:
                self.cache = json.load(f)

    def query_api(self, prompt):
        raise NotImplementedError

    def get_cache(self, prompt, instance_idx):
        sequences = self.cache.get(instance_idx, [])

        for sequence in sequences:
            if sequence.startswith(prompt) and len(sequence) > len(prompt)+1:
                return sequence
        return None

    def add_to_cache(self, sequence, instance_idx):
        if instance_idx not in self.cache:
            self.cache[instance_idx] = []
        sequences = self.cache[instance_idx]

        # newest result to the front
        sequences.append(sequence)

    def save_cache(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f)
        print('cache saved to: ' + self.cache_file)

    def get_sequence(self, prompt, instance_idx, read_cache=True):
        sequence = None
        if read_cache:
            sequence = self.get_cache(prompt, instance_idx)
        print('cached sequence')
        if sequence is None:
            print('query API')
            sequence = self.query_api(prompt)
            self.add_to_cache(sequence, instance_idx)
            #print('api sequence')
        return sequence


class OpenAI_LLM(LLM):

    def __init__(self, model_name, api_key, logit_bias=None, max_tokens=64, finish_reasons=None, **kwargs):
        openai.api_key = api_key
        self.logit_bias = logit_bias

        self.finish_reasons = finish_reasons
        if finish_reasons is None:
            self.finish_reasons = ['stop', 'length']

        super().__init__(api_key, model_name, max_tokens, **kwargs)

    def query_api(self, prompt):

        def query_func():
            return openai.Completion.create(engine=self.model_name,
                                            prompt=prompt,
                                            temperature=0,
                                            max_tokens=self.max_tokens
                                            )

        if self.logit_bias:
            def query_func():
                return openai.Completion.create(engine=self.model_name,
                                                prompt=prompt,
                                                temperature=0,
                                                max_tokens=self.max_tokens,
                                                logit_bias=self.logit_bias
                                                )

        if 'gpt-4' in self.model_name:
            def query_func():
                completion = openai.ChatCompletion.create(model=self.model_name,
                                                          messages=[{"role": "user", "content": prompt}],
                                                          max_tokens=self.max_tokens,
                                                          temperature=0,
                                                          )
                text = completion['choices'][0]['message']['content']
                if text[0] != ' ':
                    text = ' ' + text
                completion['choices'][0]['text'] = text
                return completion

        try:
            response = query_func()
        except (openai.error.ServiceUnavailableError,
                openai.error.RateLimitError,
                openai.error.ServiceUnavailableError,
                openai.error.APIConnectionError) as e:
            print(e)
            self.save_cache()
            time.sleep(10)
            print('try again')
            return self.query_api(prompt)
        print('API Response:')
        print(response)
        print('')
        self.queried_tokens += response['usage']['total_tokens']
        sequence = prompt + response['choices'][0]['text']
        assert response['choices'][0]['finish_reason'] in self.finish_reasons + [None]
        return sequence
