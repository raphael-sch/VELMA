from vln.env import get_nav_from_actions
from vln.prompt_builder import get_navigation_lines


class Agent:
    def __init__(self, query_func, env, instance, prompt_template):
        self.query_func = query_func
        self.env = env
        self.instance = instance
        self.dataset_name = instance['dataset_name']
        self.landmarks = instance['landmarks']
        self.traffic_flow = instance.get('traffic_flow')
        self.init_prompt = prompt_template.format(instance['navigation_text'])
        self.is_map2seq = instance['is_map2seq']

    def run(self, max_steps, verbatim=False):
        actions = ['init']
        if self.dataset_name == 'map2seq':
            actions.append('forward')

        navigation_lines = list()
        is_actions = list()

        query_count = 0
        nav = get_nav_from_actions(actions, self.instance, self.env)

        step_id = 0
        hints_action = None
        while step_id <= max_steps:
            if verbatim:
                print('Number of Steps:', len(nav.actions))

            new_navigation_lines, new_is_actions = get_navigation_lines(nav,
                                                                        self.env,
                                                                        self.landmarks,
                                                                        self.traffic_flow,
                                                                        step_id=step_id
                                                                        )
            navigation_lines = navigation_lines[:-1] + new_navigation_lines
            is_actions = is_actions[:-1] + new_is_actions
            step_id = len(nav.actions)

            navigation_text = '\n'.join(navigation_lines)
            prompt = self.init_prompt + navigation_text
            # print(navigation_text)

            action, queried_api, hints_action = self.query_next_action(prompt, hints_action, verbatim)
            query_count += queried_api

            action = nav.validate_action(action)

            if action == 'stop':
                nav.step(action)
                prompt += f' {action}\n'
                break

            nav.step(action)
            if verbatim:
                print('Validated action', action)

                # print('actions', actions)
                print('query_count', query_count)

        del hints_action

        new_navigation_lines, new_is_actions = get_navigation_lines(nav,
                                                                    self.env,
                                                                    self.landmarks,
                                                                    self.traffic_flow,
                                                                    step_id=step_id,
                                                                    )
        navigation_lines = navigation_lines[:-1] + new_navigation_lines
        is_actions = is_actions[:-1] + new_is_actions

        return nav, navigation_lines, is_actions, query_count

    def query_next_action(self, prompt, hints=None, verbatim=True):
        output, queried_api, hints = self.query_func(prompt, hints)
        try:
            predicted = self.extract_next_action(output, prompt)
        except Exception as e:
            print('extract_next_action error: ', e)
            print('returned "forward" instead')
            predicted_sequence = output[len(prompt):]
            predicted = 'forward'
            print('predicted_sequence', predicted_sequence)
        if verbatim:
            print('Predicted Action:', predicted)
        return predicted, queried_api, hints

    @staticmethod
    def extract_next_action(output, prompt):
        assert output.startswith(prompt)
        predicted_sequence = output[len(prompt):]
        predicted = predicted_sequence.strip().split()[0]
        predicted = predicted.lower()
        if predicted in {'forward', 'left', 'right', 'turn_around', 'stop'}:
            return predicted

        predicted = ''.join([i for i in predicted if i.isalpha()])
        if predicted == 'turn':
            next_words = predicted_sequence.strip().split()[1:]
            next_predicted = next_words[0]
            next_predicted = ''.join([i for i in next_predicted if i.isalpha()])
            next_predicted = next_predicted.lower()
            predicted += ' ' + next_predicted
        return predicted


class LLMAgent(Agent):

    def __init__(self, llm, env, instance, prompt_template):
        self.llm = llm
        self.env = env
        self.instance = instance
        self.dataset_name = instance['dataset_name']

        self.landmarks = instance['landmarks']
        self.traffic_flow = instance.get('traffic_flow')

        self.init_prompt = prompt_template.format(instance['navigation_text'])

        cache_key = f'{self.dataset_name}_{instance["idx"]}'

        def query_func(prompt, hints=None):
            queried_api = 0
            output = self.llm.get_cache(prompt, cache_key)
            if output is None:
                print('query API')
                output = self.llm.query_api(prompt)
                queried_api += 1
                self.llm.add_to_cache(output, cache_key)
                print('api sequence')
            return output, queried_api, dict()

        super().__init__(query_func, env, instance, prompt_template)
