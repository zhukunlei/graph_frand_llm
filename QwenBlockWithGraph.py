import torch
import torch.nn as nn
from transformers import Qwen3ForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast


class Qwen3WithGraphEmb(Qwen3ForCausalLM):
    def __init__(self, config, graph_emb_dim=256, fusion_layer_idx=16):
        super().__init__(config)
        self.fusion_layer_idx = fusion_layer_idx  # 第几层注入图embedding
        self.graph_proj = nn.Linear(graph_emb_dim, config.hidden_size)  # 图embedding投影到hidden_size
        self.graph_fusion_gate = nn.Linear(config.hidden_size * 2, config.hidden_size)  # 融合模块

    def inject_graph_emb(self, hidden_states, graph_emb):
        """
        hidden_states: [B, L, D]
        graph_emb: [B, D] after projection
        """
        B, L, D = hidden_states.shape
        graph_emb = graph_emb.unsqueeze(1).expand(-1, L, -1)  # [B, L, D]
        fusion_input = torch.cat([hidden_states, graph_emb], dim=-1)  # [B, L, 2D]
        fused = torch.tanh(self.graph_fusion_gate(fusion_input))  # [B, L, D]
        return hidden_states + fused  # 残差注入

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        cache_position=None,
        logits_to_keep=0,
        graph_embeddings=None,  # <=== 新增参数: 每个样本的图embedding, [B, graph_emb_dim]
        **kwargs
    ) -> CausalLMOutputWithPast:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        # 修改 BaseModel，添加中间 hook（假设 self.model 为 TransformerDecoder）
        if graph_embeddings is not None:
            projected_graph = self.graph_proj(graph_embeddings)  # [B, D]

            def hook_fn(module, input, output):
                hidden_states = output[0]  # output: (hidden_states, ...)
                fused_hidden = self.inject_graph_emb(hidden_states, projected_graph)
                return (fused_hidden,) + output[1:]  # 替换 last_hidden_state

            # 仅 hook 第 fusion_layer_idx 层
            handle = self.model.transformer.h[self.fusion_layer_idx].register_forward_hook(hook_fn)

        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )

        if graph_embeddings is not None:
            handle.remove()  # 清除hook

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
