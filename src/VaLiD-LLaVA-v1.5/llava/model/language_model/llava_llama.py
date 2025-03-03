#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig
    print("LlavaLlamaForCausalLM")
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        use_valid: Optional[bool] = None,
        valid_alpha: Optional[torch.FloatTensor] = None,
        valid_beta: Optional[torch.FloatTensor] = None,
        layer_index: Optional[int] = None, 
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,              # None                              # torch.Size([1, ])    
                position_ids,           # torch.Size([25, 635])
                attention_mask,         # torch.Size([25, 635])
                past_key_values,        # None
                inputs_embeds,          # torch.Size([25, 635, 4096])       # None
                labels                  # None
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images, 
                layer_index=layer_index
            )
        
        """
        # print("=> llava_llama:", output_attentions, output_hidden_states, return_dict)
        # :: 这里的super().forward()函数肯定是上级被继承class中的forward函数，
        # :: 在def __init__()下的super(LlamaForCausalLM, self).__init__(config)得知，父类函数是LlamaForCausalLM
        # :: 也就是说明这里的forward函数是transformer库函数中的LlamaForCausalLM，具体位置在modeling_llama.py中
        # :: 相应地我们也应该把不同层的视觉特征传递到modelling_llama函数中，再由modelling_llama传递到generation.utils中进行sampling
        # 这里inputs_embeds的维度取决于clip的层数，越大占用的显存越多，等效于layer_num张图片同时进行推理
        # 而且这里不能改，因为涉及前传过程，除非把logits计算的部分拿到下面写了
        
        for name, ele in zip(["input_ids", "position_ids", "attention_mask", "past_key_values", "inputs_embeds", "labels"], [input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels]):
            if ele is None:
                print(f"{name}  =>  None")
            else:
                if isinstance(ele, tuple):
                    if len(tuple) == 1:
                        for index, subele in enumerate(ele[0]):
                            print(f"{name}-{index}  =>  {subele.shape}")
                print(f"{name}  =>  {ele.shape}")
        print("*"*100+"\n\n")
        """
        
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, 
            labels=labels,
            use_cache=use_cache,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True
        )
    
    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        _inputs.update({"layer_index": kwargs.get('layer_index')}) 
        if images is not None:
            _inputs['images'] = images
        return _inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
