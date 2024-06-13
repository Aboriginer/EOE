class PromptGenerator:
    def __init__(self):
        self.dataset_info = {
            # class_type is only used in the fine-grained OOD task
            # task: far and near
            'bird200': {'class_type': None, 'num_classes': 200},
            'car196': {'class_type': None, 'num_classes': 196},
            'food101': {'class_type': None, 'num_classes': 101},
            'pet37': {'class_type': None, 'num_classes': 37},
            'ImageNet10': {'class_type': None, 'num_classes': 10},
            'ImageNet20': {'class_type': None, 'num_classes': 20},
            'ImageNet': {'class_type': None, 'num_classes': 1000},
            'cifar10': {'class_type': None, 'num_classes': 10},
            'cifar100': {'class_type': None, 'num_classes': 100},

            # task: fine-grained
            'cub100_ID': {'class_type': 'bird', 'num_classes': 100},
            'car98_ID': {'class_type': 'car', 'num_classes': 98},
            'pet18_ID': {'class_type': 'pet(including dogs and cats)', 'num_classes': 18},
            'food50_ID': {'class_type': 'food', 'num_classes': 50},
        }

    def _fine_grained_prompt(self, class_type, num_classes, class_info, envision_nums=50):
        return f"""Q: I have a dataset containing 10 unique species of dogs. I need a list of 10 distinct dog species that are NOT present in my dataset, and ensure there are no repetitions in the list you provide. For context, the species in my dataset are: ['husky dog', 'alaskan Malamute', 'cossack sled dog', 'golden retriever', 'German Shepherd', 'Beagle', 'Bulldog', 'Poodle', 'Dachshund', 'Doberman Pinscher']
A: The other 10 dog species not in the dataset are:
- Labrador Retriever
- Rottweiler
- Boxer
- Border Collie
- Shih Tzu
- Akita
- Saint Bernard
- Australian Shepherd
- Great Dane
- Boston Terrier

Q: I have a dataset containing {num_classes} different species of {class_type}. I need a list of {envision_nums} distinct {class_type} species that are NOT present in my dataset, and ensure there are no repetitions in the list you provide. For context, the species in my dataset are: {class_info}
A: The other {envision_nums} {class_type} species not in the dataset are:
"""

    def _fine_grained_irrelevant_prompt(self, class_type, num_classes, class_info, envision_nums=50):
        return f"""Q: I have a dataset containing 10 unique species of dogs. I need a list of 10 distinct categories that are irrelevant to my dataset's categories. These categories you gave me should NOT be present in my dataset, and ensure there are no repetitions in the list you provide. For context, the species in my dataset are: ['husky dog', 'alaskan Malamute', 'cossack sled dog', 'golden retriever', 'German Shepherd', 'Beagle', 'Bulldog', 'Poodle', 'Dachshund', 'Doberman Pinscher']
A: The 10 distinct irrelevant categories are:
- Tropical Fish
- Houseplants
- Mountain Bikes
- Computer Software
- Solar Panels
- Classic Novels
- Electronic Keyboards
- Sports Cars
- Kitchen Appliances
- Board Games

Q: I have a dataset containing {num_classes} different species of {class_type}. I need a list of {envision_nums} distinct categories that are irrelevant to my dataset's categories. These categories you gave me should NOT be present in my dataset, and ensure there are no repetitions in the list you provide. For context, the species in my dataset are: {class_info}
A: The {envision_nums} distinct irrelevant categories are:
"""
    
    def _fine_grained_dissimilar_prompt(self, class_type, num_classes, class_info, envision_nums=50):
        return f"""Q: I have a dataset containing 10 unique species of dogs. I need a list of 10 distinct categories visually dissimilar from my dataset's categories. These categories you gave me should NOT be present in my dataset, and ensure there are no repetitions in the list you provide. For context, the species in my dataset are: ['husky dog', 'alaskan Malamute', 'cossack sled dog', 'golden retriever', 'German Shepherd', 'Beagle', 'Bulldog', 'Poodle', 'Dachshund', 'Doberman Pinscher']
A: The 10 distinct visually dissimilar categories are:
- Desert cacti
- Deep sea creatures
- Exotic birds
- Space telescopes
- High-speed trains
- Medieval castles
- Microscopic organisms
- Mountain landscapes
- Formula One cars
- Ancient Egyptian artifacts

Q: I have a dataset containing {num_classes} different species of {class_type}. I need a list of {envision_nums} distinct categories visually dissimilar from my dataset's categories. These categories you gave me should NOT be present in my dataset, and ensure there are no repetitions in the list you provide. For context, the species in my dataset are: {class_info}
A: The {envision_nums} distinct visually dissimilar categories are:
"""

    def _near_prompt(self, class_type, num_classes, class_info, envision_nums=3):
        return f"""Q: Given the image category [husky dog], please suggest visually similar categories that are not directly related or belong to the same primary group as [husky dog]. Provide suggestions that share visual characteristics but are from broader and different domains than [husky dog].
A: There are 3 classes similar to [husky dog], and they are from broader and different domains than [husky dog]:
- gray wolf
- black stone
- red panda

Q: Given the image category [basketball], please suggest visually similar categories that are not directly related or belong to the same primary group as [basketball]. Provide suggestions that share visual characteristics but are from broader and different domains than [basketball].
A: There are 3 classes similar to [basketball], and they are from broader and different domains than [basketball]:
- balloons
- blowfish
- hat

Q: Given the image category [water jug], please suggest visually similar categories that are not directly related or belong to the same primary group as [water jug]. Provide suggestions that share visual characteristics but are from broader and different domains than [water jug].
A: There are 3 classes similar to [water jug], and they are from broader and different domains than [water jug]:
- trumpets
- helmets
- rucksacks

Q: Given the image category [{class_info}], please suggest visually similar categories that are not directly related or belong to the same primary group as [{class_info}]. Provide suggestions that share visual characteristics but are from broader and different domains than [{class_info}].
A: There are {envision_nums} classes similar to [{class_info}], and they are from broader and different domains than [{class_info}]:
"""
    
    def _near_irrelevant_prompt(self, class_type, num_classes, class_info, envision_nums=3):
        return f"""Q: Given the image category [husky dog], please suggest categories that are visually irrelevant to [husky dog]. Provide suggestions that are from broader and different domains than [husky dog].
A: There are 3 categories visually irrelevant to [husky dog], and they are from broader and different domains than [husky dog]:
- urban landscapes
- astronomical phenomena
- historical artifacts

Q: Given the image category [basketball], please suggest categories that are visually irrelevant to [basketball]. Provide suggestions that are from broader and different domains than [basketball].
A: There are 3 categories visually irrelevant to [basketball], and they are from broader and different domains than [basketball]:
- marine life
- architectural landmarks
- medieval art

Q: Given the image category [water jug], please suggest categories that are visually irrelevant to [water jug]. Provide suggestions that are from broader and different domains than [water jug].
A: There are 3 categories visually irrelevant to [water jug], and they are from broader and different domains than [water jug]:
- wildlife photography
- ancient ruins
- sky

Q: Given the image category [{class_info}], please suggest categories that are visually irrelevant to [{class_info}]. Provide suggestions that are from broader and different domains than [{class_info}].
A: There are {envision_nums} categories visually irrelevant to [{class_info}], and they are from broader and different domains than [{class_info}]:
"""

    def _near_dissimilar_prompt(self, class_type, num_classes, class_info, envision_nums=3):
        return f"""Q: Given the image category [husky dog], please suggest categories that are visually dissimilar to [husky dog]. Provide suggestions that do not share visual characteristics but are from broader and different domains than [husky dog].
A: There are 3 categories visually dissimilar from [husky dog], and they are from broader and different domains than [husky dog]:
- skyscrapers
- reefs
- space nebulae

Q: Given the image category [basketball], please suggest categories that are visually dissimilar to [basketball]. Provide suggestions that do not share visual characteristics but are from broader and different domains than [basketball].
A: There are 3 categories visually dissimilar from [basketball], and they are from broader and different domains than [basketball]:
- vehicles
- desert
- ancient ruins

Q: Given the image category [water jug], please suggest categories that are visually dissimilar to  [water jug]. Provide suggestions that do not share visual characteristics but are from broader and different domains than [water jug].
A: There are 3 categories visually dissimilar from [water jug], and they are from broader and different domains than [water jug]:
- medieval castles
- microscopic cells
- sky

Q: Given the image category [{class_info}], please suggest categories that are visually dissimilar to  [{class_info}]. Provide suggestions that do not share visual characteristics but are from broader and different domains than [{class_info}].
A: There are {envision_nums} categories visually dissimilar from [{class_info}], and they are from broader and different domains than [{class_info}]:
"""

    def _far_prompt(self, class_type, num_classes, class_info, envision_nums=50): 
       return f"""Q: I have gathered images of 4 distinct categories: ['Husky dog', 'Garfield cat', 'churches', 'truck']. Summarize what broad categories these categories might fall into based on visual features. Now, I am looking to identify 5 categories that visually resemble to these broad categories but have no direct relation to these broad categories. Please list these 5 items for me.
A: These 5 items are:
- black stone
- mountain
- Ginkgo Tree
- river
- Rapeseed

Q: I have gathered images of {num_classes} distinct categories: [{class_info}]. Summarize what broad categories these categories might fall into based on visual features. Now, I am looking to identify {envision_nums} classes that visually resemble to these broad categories but have no direct relation to these broad categories. Please list these {envision_nums} items for me.
A: These {envision_nums} items are:
"""
    
    def _far_irrelevant_prompt(self, class_type, num_classes, class_info, envision_nums=50): 
       return f"""Q: I have gathered images of 4 distinct categories: ['Husky dog', 'Garfield cat', 'churches', 'truck']. Summarize what broad categories these categories might fall into based on visual features. Now, I am looking to identify 5 categories that are visually irrelevant to these broad categories. Please list these 5 items for me.
A: These 5 items are:
- space phenomena
- underwater landscapes
- abstract art
- river
- microscopic organisms

Q: I have gathered images of {num_classes} distinct categories: [{class_info}]. Summarize what broad categories these categories might fall into based on visual features. Now, I am looking to identify {envision_nums}  classes that are visually irrelevant to these broad categories. Please list these {envision_nums}  items for me.
A: These {envision_nums}  items are:
"""
    
    def _far_dissimilar_prompt(self, class_type, num_classes, class_info, envision_nums=50): 
       return f"""Q: I have gathered images of 4 distinct categories: ['Husky dog', 'Garfield cat', 'churches', 'truck']. Summarize what broad categories these categories might fall into based on visual features. Now, I am looking to identify 5 categories that are visually dissimilar from these broad categories. Please list these 5 items for me.
A: These 5 items are:
- tropical fish
- computer hardware
- space phenomena
- desert landscapes
- abstract art

Q: I have gathered images of {num_classes} distinct categories: [{class_info}]. Summarize what broad categories these categories might fall into based on visual features. Now, I am looking to identify {envision_nums} classes that are visually dissimilar from these broad categories. Please list these {envision_nums} items for me.
A: These {envision_nums} items are:
"""

    def get_prompt(self, ood_task, in_dataset, class_info=None, envision_nums=50):
        '''
        Args:
        - ood_task: choices=['fine_grained', 'near', 'far']
        - in_dataset: ID dataset
        - class_info: for ood_task in ['far'], class_info=None
                      for ood_task in ['fine_grained'], class_info is total class_name list
                      for ood_task in ['near'], class_info is single class_name
        '''
        prompt_dispatcher = {
            'fine_grained': self._fine_grained_prompt,
            'fine_grained_irrelevant': self._fine_grained_irrelevant_prompt,
            'fine_grained_dissimilar': self._fine_grained_dissimilar_prompt,
            'near': self._near_prompt,
            'near_irrelevant': self._near_irrelevant_prompt,
            'near_dissimilar': self._near_dissimilar_prompt,
            'far': self._far_prompt,
            'far_irrelevant': self._far_irrelevant_prompt,
            'far_dissimilar': self._far_dissimilar_prompt,
            'general': self._far_prompt,
        }

        if ood_task not in prompt_dispatcher:
            raise ValueError("Unknown ood task!")

        dataset_info = self.dataset_info[in_dataset]
        class_type = dataset_info["class_type"]
        num_classes = dataset_info["num_classes"]
        return prompt_dispatcher[ood_task](class_type, num_classes, class_info, envision_nums)
    
    def _fine_grained_prompt_again(self, in_dataset, envision_nums=50):
        dataset_info = self.dataset_info[in_dataset]
        class_type = dataset_info["class_type"]
        num_classes = dataset_info["num_classes"]
        return f"""Q: Provide {envision_nums} additional {class_type} categories that aren't in the set I gave you, and haven't been mentioned in your previous responses to me.
A: The {envision_nums} additional categories are:
"""

    def _fine_grained_irrelevant_prompt_again(self, in_dataset, envision_nums=50):
        dataset_info = self.dataset_info[in_dataset]
        class_type = dataset_info["class_type"]
        num_classes = dataset_info["num_classes"]
        return f"""Q: Provide {envision_nums} additional distinct categories that are irrelevant to my dataset's categories, and haven't been mentioned in your previous responses to me.
A: The {envision_nums} additional categories are:
"""
    
    def _fine_grained_dissimilar_prompt_again(self, in_dataset, envision_nums=50):
        dataset_info = self.dataset_info[in_dataset]
        class_type = dataset_info["class_type"]
        num_classes = dataset_info["num_classes"]

        return f"""Q: Provide {envision_nums} additional distinct categories that are visually dissimilar from my dataset's categories, and haven't been mentioned in your previous responses to me.
A: The {envision_nums} additional categories are:
"""

    def _far_prompt_again(self, in_dataset, envision_nums=50):
        return f"""Q: Give me {envision_nums} more categories that are visually similar to these broad categories you summarized in the dataset but have no direct relation to these broad categories. Each category you give cannot exceed three words and could not have appeared in your previous answers.
A: The other {envision_nums} categories are:
"""
    
    def _far_irrelevant_prompt_again(self, in_dataset, envision_nums=50):
        return f"""Q: Give me {envision_nums} more categories that are visually irrelevant to these broad categories you summarized in the dataset. Each category you give cannot exceed three words and could not have appeared in your previous answers.
A: The other {envision_nums} categories are:
"""
    
    def _far_dissimilar_prompt_again(self, in_dataset, envision_nums=50):
        return f"""Q: Give me {envision_nums} more categories that are visually dissimilar from these broad categories you summarized in the dataset. Each category you give cannot exceed three words and could not have appeared in your previous answers.
A: The other {envision_nums} categories are:
"""

    def get_prompt_again(self, ood_task, in_dataset, class_info=None, envision_nums=50):
        '''
        Args:
        - ood_task: choices=['fine_grained', 'near', 'far']
        - in_dataset: ID dataset
        - class_info: for ood_task in ['far'], class_info=None
                      for ood_task in ['fine_grained'], class_info is total class_name list
                      for ood_task in ['near'], class_info is single class_name
        '''
        prompt_dispatcher = {
            'fine_grained': self._fine_grained_prompt_again,
            'fine_grained_irrelevant': self._fine_grained_irrelevant_prompt_again,
            'fine_grained_dissimilar': self._fine_grained_dissimilar_prompt_again,
            'far': self._far_prompt_again,
            'far_irrelevant': self._far_irrelevant_prompt_again,
            'far_dissimilar': self._far_dissimilar_prompt_again,
            'general': self._far_prompt_again,
        }

        if ood_task not in prompt_dispatcher:
            raise ValueError("Unknown ood task!")

        return prompt_dispatcher[ood_task](in_dataset, envision_nums)