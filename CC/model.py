from re import A
import torch
import torch.nn as nn
from transformers import BertConfig, BertModel
from CC.LEBert import WCBertModel, BertPreTrainedModel
from CC.ZLEBert import ZWCBertModel
from CC.PCBert import PCBertModel
from CC.ZLEBert_v2 import ZWCBertModel_v2
from CC.ZLEBert_v3 import ZWCBertModel_v3
from CC.crf import CRF
from CC.birnncrf import BiRnnCrf
from ICCSupervised.ICCSupervised import IModel


class CCNERModel(IModel):

    def __init__(self, **args):
        required_pretrained_embedding_models = ['LEBert', 'LEBertFusion', 'PLEBert', 'ZLEBert', 'ZLEBert_v2','ZLEBert_v3']
        required_label_embedding_models = ['PLEBert']
        required_inter_embedding_models = ['ZLEBert','ZLEBert_v2','ZLEBert_v3']
        assert "model_name" in args, "argument model_name required"
        assert "bert_config_file_name" in args, "argument bert_config_file_name required"
        assert "pretrained_file_name" in args, "argument pretrained_file_name required"
        assert "tagset_size" in args, "argument tagset_size required"
        assert "hidden_dim" in args, "argument hidden_dim required"
        self.model_name = args['model_name']
        self.bert_config_file_name = args['bert_config_file_name']
        self.pretrained_file_name = args['pretrained_file_name']
        self.tagset_size = args['tagset_size']
        self.hidden_dim = args['hidden_dim']
        if args['model_name'] in required_pretrained_embedding_models:
            assert "pretrained_embeddings" in args, "argument pretrained_embeddings required"
            self.pretrained_embeddings = args['pretrained_embeddings']
        if args['model_name'] in required_label_embedding_models:
            assert "label_embeddings" in args, "argument label_embeddings required"
            self.label_embeddings = args['label_embeddings']

        if args['model_name'] in required_inter_embedding_models:
            assert "inter_embeddings" in args, "argument inter_embedding required"
            self.inter_embeddings = args['inter_embeddings']
        
        self.load_model()

    def load_model(self):
        config = BertConfig.from_json_file(self.bert_config_file_name)
        if self.model_name == 'LEBert':
            self.model = LEBertModel.from_pretrained(
            self.pretrained_file_name, pretrained_embeddings=self.pretrained_embeddings, config=config)
        elif self.model_name == 'ZLEBert_v3':
            self.model = ZLEBertModel_v3.from_pretrained(
            self.pretrained_file_name, pretrained_embeddings=self.pretrained_embeddings, config=config, inter_embeddings = self.inter_embeddings)
        elif self.model_name == 'ZLEBert_v2':
            self.model = ZLEBertModel_v2.from_pretrained(
            self.pretrained_file_name, pretrained_embeddings=self.pretrained_embeddings, config=config, inter_embeddings = self.inter_embeddings)
        elif self.model_name == 'ZLEBert':
            self.model = ZLEBertModel.from_pretrained(
            self.pretrained_file_name, pretrained_embeddings=self.pretrained_embeddings, config=config, inter_embeddings = self.inter_embeddings)
        elif self.model_name == 'LEBertFusion':
            self.model = LEBertModelFusion.from_pretrained(
            self.pretrained_file_name, pretrained_embeddings=self.pretrained_embeddings, config=config)
        elif self.model_name == 'PLEBert':
            self.model = PLEBertModel.from_pretrained(
            self.pretrained_file_name, pretrained_embeddings=self.pretrained_embeddings, config=config)
        elif self.model_name == 'Bert':
            self.model = BertBaseModel.from_pretrained(
            self.pretrained_file_name, config=config)
        self.birnncrf = BiRnnCrf(
            tagset_size=self.tagset_size, embedding_dim=config.hidden_size, hidden_dim=self.hidden_dim)

    def get_model(self):
        return self.model, self.birnncrf

    def __call__(self):
        return self.get_model()

class BertBaseModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

        self.init_weights()
    
    def forward(self, **args):
        input = {
            'input_ids': args['input_ids'],
            'attention_mask': args['attention_mask']
        }
        if 'token_type_ids' in args:
            input['token_type_ids'] = args['token_type_ids']
        outputs = self.bert(**input)
        return {
            'mix_output': outputs.last_hidden_state,
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }
# inter
class ZLEBertModel_v3(BertPreTrainedModel):
    '''
    config: BertConfig
    pretrained_embeddings: 预训练embeddings shape: size * 200
    '''

    def __init__(self, config, pretrained_embeddings, inter_embeddings):
        super().__init__(config)

        word_vocab_size = pretrained_embeddings.shape[0]
        embed_dim = pretrained_embeddings.shape[1]
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_dim)
        
        inter_word_vocab_size = inter_embeddings.shape[0]
        inter_embed_dim = inter_embeddings.shape[1]
        self.inter_word_embeddings = nn.Embedding(inter_word_vocab_size, inter_embed_dim)
        
        self.bert = ZWCBertModel_v3(config)

        self.init_weights()
        
        # self.inter_word_embeddings.weight.data.copy_(
        #     torch.from_numpy(inter_embeddings))
        # init the embedding
        self.word_embeddings.weight.data.copy_(
            torch.from_numpy(pretrained_embeddings))
        print("Load pretrained embedding from file.........")

    def forward(
            self,
            **args
    ):

        matched_word_embeddings = self.word_embeddings(
            args['matched_word_ids'])   
        matched_inter_embeddings = self.inter_word_embeddings(
            args['inter_matched_word_ids'])
        outputs = self.bert(
            input_ids=args['input_ids'],
            attention_mask=args['attention_mask'],
            token_type_ids=args['token_type_ids'],
            matched_word_embeddings=matched_word_embeddings,
            matched_word_mask=args['matched_word_mask'],

            matched_inter_embeddings= matched_inter_embeddings,
            matched_inter_mask=args['inter_matched_word_mask'],
        )

        sequence_output = outputs[0]

        return {
            'mix_output': sequence_output,
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }

# inter
class ZLEBertModel_v2(BertPreTrainedModel):
    '''
    config: BertConfig
    pretrained_embeddings: 预训练embeddings shape: size * 200
    '''

    def __init__(self, config, pretrained_embeddings, inter_embeddings):
        super().__init__(config)

        word_vocab_size = pretrained_embeddings.shape[0]
        embed_dim = pretrained_embeddings.shape[1]
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_dim)
        
        inter_word_vocab_size = inter_embeddings.shape[0]
        inter_embed_dim = inter_embeddings.shape[1]
        self.inter_word_embeddings = nn.Embedding(inter_word_vocab_size, inter_embed_dim)
        
        self.bert = ZWCBertModel_v2(config)

        self.init_weights()
        
        # self.inter_word_embeddings.weight.data.copy_(
        #     torch.from_numpy(inter_embeddings))
        # init the embedding
        self.word_embeddings.weight.data.copy_(
            torch.from_numpy(pretrained_embeddings))
        print("Load pretrained embedding from file.........")

    def forward(
            self,
            **args
    ):


        matched_word_embeddings = self.word_embeddings(
            args['matched_word_ids'])   
        matched_inter_embeddings = self.inter_word_embeddings(
            args['inter_matched_word_ids'])
        outputs = self.bert(
            input_ids=args['input_ids'],
            attention_mask=args['attention_mask'],
            token_type_ids=args['token_type_ids'],
            matched_word_embeddings=matched_word_embeddings,
            matched_word_mask=args['matched_word_mask'],

            matched_inter_embeddings= matched_inter_embeddings,
            matched_inter_mask=args['inter_matched_word_mask'],
        )

        sequence_output = outputs[0]

        return {
            'mix_output': sequence_output,
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }
# inter
class ZLEBertModel(BertPreTrainedModel):
    '''
    config: BertConfig
    pretrained_embeddings: 预训练embeddings shape: size * 200
    '''

    def __init__(self, config, pretrained_embeddings, inter_embeddings):
        super().__init__(config)

        word_vocab_size = pretrained_embeddings.shape[0]
        embed_dim = pretrained_embeddings.shape[1]
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_dim)
        
        inter_word_vocab_size = inter_embeddings.shape[0]
        inter_embed_dim = inter_embeddings.shape[1]
        self.inter_word_embeddings = nn.Embedding(inter_word_vocab_size, inter_embed_dim)
        
        self.bert = ZWCBertModel(config)

        self.init_weights()
        
        # self.inter_word_embeddings.weight.data.copy_(
        #     torch.from_numpy(inter_embeddings))
        # init the embedding
        self.word_embeddings.weight.data.copy_(
            torch.from_numpy(pretrained_embeddings))
        print("Load pretrained embedding from file.........")

    def forward(
            self,
            **args
    ):
        # print(args['matched_word_ids'].shape)
        # print(1)
        # print(args['inter_matched_word_ids'].shape)
        # print(2)
        # print(0/0)

        matched_word_embeddings = self.word_embeddings(
            args['matched_word_ids'])   
        matched_inter_embeddings = self.inter_word_embeddings(
            args['inter_matched_word_ids'])
        outputs = self.bert(
            input_ids=args['input_ids'],
            attention_mask=args['attention_mask'],
            token_type_ids=args['token_type_ids'],
            matched_word_embeddings=matched_word_embeddings,
            matched_word_mask=args['matched_word_mask'],

            matched_inter_embeddings= matched_inter_embeddings,
            matched_inter_mask=args['inter_matched_word_mask'],
        )

        sequence_output = outputs[0]

        return {
            'mix_output': sequence_output,
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }


class LEBertModel(BertPreTrainedModel):
    '''
    config: BertConfig
    pretrained_embeddings: 预训练embeddings shape: size * 200
    '''

    def __init__(self, config, pretrained_embeddings):
        super().__init__(config)

        word_vocab_size = pretrained_embeddings.shape[0]
        embed_dim = pretrained_embeddings.shape[1]
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_dim)
        self.bert = WCBertModel(config)

        self.init_weights()

        # init the embedding
        self.word_embeddings.weight.data.copy_(
            torch.from_numpy(pretrained_embeddings))
        print("Load pretrained embedding from file.........")

    def forward(
            self,
            **args
    ):
        matched_word_embeddings = self.word_embeddings(
            args['matched_word_ids'])
        outputs = self.bert(
            input_ids=args['input_ids'],
            attention_mask=args['attention_mask'],
            token_type_ids=args['token_type_ids'],
            matched_word_embeddings=matched_word_embeddings,
            matched_word_mask=args['matched_word_mask']
        )

        sequence_output = outputs[0]

        return {
            'mix_output': sequence_output,
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }


class LEBertModelFusion(BertPreTrainedModel):
    '''
    config: BertConfig
    pretrained_embeddings: 预训练embeddings shape: size * 200
    '''

    def __init__(self, config, pretrained_embeddings):
        super().__init__(config)

        word_vocab_size = pretrained_embeddings.shape[0]
        embed_dim = pretrained_embeddings.shape[1]
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_dim)
        self.bert = WCBertModel(config)

        self.init_weights()

        self.word_transform = nn.Linear(
            config.word_embed_dim, config.hidden_size)
        self.act = nn.Tanh()
        self.word_word_weight = nn.Linear(
            config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.HP_dropout)

        attn_W = torch.zeros(config.hidden_size, config.hidden_size)
        self.attn_W = nn.Parameter(attn_W)
        self.attn_W.data.normal_(mean=0.0, std=config.initializer_range)

        # init the embedding
        self.word_embeddings.weight.data.copy_(
            torch.from_numpy(pretrained_embeddings))
        print("Load pretrained embedding from file.........")

    def forward(
            self,
            **args
    ):
        matched_word_embeddings = self.word_embeddings(
            args['matched_word_ids'])
        outputs = self.bert(
            input_ids=args['input_ids'],
            attention_mask=args['attention_mask'],
            token_type_ids=args['token_type_ids'],
            matched_word_embeddings=matched_word_embeddings,
            matched_word_mask=args['matched_word_mask']
        )

        sequence_output = outputs[0]

        matched_word_embeddings = self.word_transform(matched_word_embeddings)
        matched_word_embeddings = self.act(matched_word_embeddings)
        matched_word_embeddings = self.word_word_weight(
            matched_word_embeddings)
        matched_word_embeddings = self.dropout(matched_word_embeddings)

        alpha = torch.matmul(sequence_output.unsqueeze(2),
                             self.attn_W)  # [N, L, 1, D]
        alpha = torch.matmul(alpha, torch.transpose(
            matched_word_embeddings, 2, 3))  # [N, L, 1, W]
        alpha = alpha.squeeze()  # [N, L, W]
        alpha = alpha + \
            (1 - args['matched_word_mask'].float()) * (-2 ** 31 + 1)
        alpha = torch.nn.Softmax(dim=-1)(alpha)  # [N, L, W]
        alpha = alpha.unsqueeze(-1)  # [N, L, W, 1]
        matched_word_embeddings = torch.sum(
            matched_word_embeddings * alpha, dim=2)  # [N, L, D]

        # concat the embedding [B, L, N, D], [B, L, N]
        sequence_output = torch.cat(
            (sequence_output, matched_word_embeddings), dim=-1)
        sequence_output = self.dropout(sequence_output)

        return {
            'mix_output': sequence_output,
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }

class PLEBertModel(BertPreTrainedModel):
    '''
    config: BertConfig
    pretrained_embeddings: 预训练embeddings shape: size * 200
    '''

    def __init__(self, config, pretrained_embeddings):
        super().__init__(config)

        word_vocab_size = pretrained_embeddings.shape[0]
        embed_dim = pretrained_embeddings.shape[1]
        self.word_embeddings = nn.Embedding(word_vocab_size, embed_dim)

        # label_vocab_size = label_embeddings.shape[0]
        # label_dim = label_embeddings.shape[1]
        # self.label_embeddings = nn.Embedding(label_vocab_size, label_dim)
        self.bert = PCBertModel(config)

        self.init_weights()

        # init the embedding
        self.word_embeddings.weight.data.copy_(
            torch.from_numpy(pretrained_embeddings))
        # self.label_embeddings.weight.data.copy_(
        #     torch.from_numpy(label_embeddings))
        print("Load pretrained embedding from file.........")

    def forward(
            self,
            **args
    ):
        matched_word_embeddings = self.word_embeddings(
            args['matched_word_ids'])
        # matched_label_embeddings = self.label_embeddings(
        #     args['matched_label_ids'])
        outputs = self.bert(
            input_ids=args['input_ids'],
            attention_mask=args['attention_mask'],
            token_type_ids=args['token_type_ids'],
            matched_word_embeddings=matched_word_embeddings,
            matched_label_embeddings=args['matched_label_embeddings'],
            matched_word_mask=args['matched_word_mask'],
            matched_label_mask=args['matched_label_mask']
        )

        sequence_output = outputs[0]

        return {
            'mix_output': sequence_output,
            'last_hidden_state': outputs.last_hidden_state,
            'pooler_output': outputs.pooler_output,
            'hidden_states': outputs.hidden_states,
            'attentions': outputs.attentions,
        }