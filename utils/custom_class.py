# still use small llama
import torch
from torch import nn
from typing import List, Optional, Tuple, Union
from transformers import LlamaModel, LlamaPreTrainedModel
from transformers.cache_utils import Cache
class LlamaForMultiClassTokenClassification(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)  # Pretrained Llama model (LlamaForCausalLM)
        self.num_classes = config.num_labels
        vocab_size = config.vocab_size

        # Project hidden states to logits of shape [vocab_size * num_classes]
        self.score = nn.Linear(config.hidden_size, vocab_size * self.num_classes)
        self.post_init()

    # def forward(
    #     self,
    #     input_ids: Optional[torch.LongTensor] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    #     position_ids: Optional[torch.LongTensor] = None,
    #     past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    #     inputs_embeds: Optional[torch.FloatTensor] = None,
    #     labels: Optional[torch.LongTensor] = None,
    #     class_labels: Optional[torch.LongTensor] = None,
    #     prompt_len: Optional[torch.LongTensor] = None,
    #     use_cache: Optional[bool] = None,
    #     output_attentions: Optional[bool] = None,
    #     output_hidden_states: Optional[bool] = None,
    #     return_dict: Optional[bool] = None,
    #     ) -> Union[Tuple, MultiClassTokenClassificationOutputWithPast]:
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        prompt_len: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
        
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs[0]
        
        
        # Project to vocab_size * num_classes for token-level prediction
        logits = self.score(last_hidden_state)
        batch_size, seq_len, _ = logits.shape
        logits = logits.view(batch_size, seq_len, -1, self.num_classes)
        batch_size, seq_len, _, _ = logits.shape
        loss = None
        masked_class_labels = None
        if labels is None:
            if not return_dict:
                return logits
            else:
                return MultiClassTokenClassificationOutputWithPast(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            past_key_values=outputs.past_key_values,
                )
            
        shift_logits = logits[:, :-1, :, :].contiguous()  # Shape: [batch_size, seq_len-1, vocab_size, num_classes]
        shift_labels = labels[:, 1:].contiguous()  # Shifted labels for token prediction (Shape: [batch_size, seq_len-1])
        shift_attention_mask = attention_mask[:, 1:].contiguous()  # Shifted attention mask
        text_len = torch.sum(shift_attention_mask, dim = 1)
        text_start = seq_len - 1 - text_len + prompt_len
        token_indices = torch.arange(seq_len - 1).unsqueeze(0).to(input_ids.device)
        mask = (token_indices >= text_start.unsqueeze(1)) 
        masked_logits = shift_logits[mask]
        masked_labels = shift_labels[mask]
        extracted_logits = masked_logits[torch.arange(masked_logits.size(0)), masked_labels]
        del masked_logits, masked_labels
        loss_fn = nn.CrossEntropyLoss(reduction='none')  # We use 'none' to get individual losses per class
        if class_labels is not None:
            class_labels_expanded = class_labels.unsqueeze(1).expand(batch_size, seq_len - 1)
            masked_class_labels = class_labels_expanded[mask]
            length = mask.sum(dim = 1).cumsum(dim = 0)
            length = torch.cat((torch.tensor([0]).to(length.device),length), dim=0)
            loss = loss_fn(extracted_logits, masked_class_labels)
            batched_loss = []
            for i in range(batch_size):
                batched_loss.append(loss[length[i]:length[i+1]].mean())
            
            
            loss = torch.stack(batched_loss).mean()


            batched_acc = []
            predicted_class = extracted_logits.argmax(dim=-1)  # Shape: [number of tokens]
            correct_predictions = (predicted_class == masked_class_labels).float()
            for i in range(batch_size):
                batched_acc.append(correct_predictions[length[i]:length[i+1]].mean())
            average_acc = torch.stack(batched_acc).mean()  # Mean of correct predictions as accuracy
            
            
            
            
  
        if not return_dict:
            if masked_class_labels is not None:
                output = (extracted_logits, masked_class_labels, logits)+ outputs[1:]
            else:
                output = (extracted_logits, logits)+ outputs[1:]
            return ((loss,) + output) if loss is not None else output
        else:
            return MultiClassTokenClassificationOutputWithPast(
                loss = loss,
                logits = logits,
                average_acc = average_acc,
                hidden_states = outputs.hidden_states,
                attentions = outputs.attentions,
                # past_key_values=outputs.past_key_values, 
                )

            
                

        

from transformers.utils import ModelOutput
from typing import Optional, Tuple
import torch
from dataclasses import dataclass
@dataclass
class MultiClassTokenClassificationOutputWithPast(ModelOutput):
    """
    Custom output class for multi-class token classification tasks, with additional outputs for
    past key values, accuracies, and loss over the last token.
    """
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    average_acc: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[torch.FloatTensor]] = None  # For autoregressive models



