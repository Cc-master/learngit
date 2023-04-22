# -*- coding: utf-8 -*-
# @Author : wgq
# @time   : 2023/1/5
# @File   : json_open.py
# Software: PyCharm
# explain: 语料文件读取  并生成新的语料

import argparse
import collections
import json
import os
from log import logger
from typing import Optional
import hashlib
from dataclasses import dataclass, field

import torch
import torch.distributed
from transformers import BertModel, BertTokenizer, AutoModelForMaskedLM

from tqdm import tqdm
from transformers.data.processors.squad import SquadV1Processor, TensorDataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TFAutoModelForQuestionAnswering,
    TFTrainer,
    TFTrainingArguments,
    squad_convert_examples_to_features,
)
from transformers.models.bert import tokenization_bert


def _check_is_max_context(doc_spans, cur_span_index, position):
    """计算每个token在每一段滑窗中的最佳位置
    maximum context分数：上下文最小值"""
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index
    return cur_span_index == best_span_index


def md5vale(str):
    input_name = hashlib.md5()
    input_name.update(str.encode("utf-8"))
    return input_name.hexdigest()





def read_squad_examples(input_file, is_training, version_2_with_negative):
    """
    读入数据
    :param input_file:数据文件路径
    :param is_training:是否为训练数据集，若是将会读取answers部分，否则将该部分给出为None
    :param version_2_with_negative:是否含有拒识问题
    :return:每一个example的SquadExample实例组成的list
    """
    pages = 0
    with open(input_file, 'r', encoding='utf-8') as reader:
        input_data = json.load(reader)['data']

    # def is_whitespace(c):
    #     if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
    #         return True
    #     return False

    examples = []
    for entry in input_data:
        for paragraph in entry['paragraphs']:
            paragraph_text = paragraph['context']

            doc_tokens = []
            char_to_word_offset = []
            #             prev_is_whitespace = True
            for c in paragraph_text:
                #                 if is_whitespace(c):
                #                     prev_is_whitespace = True
                #                 else:
                #                     if prev_is_whitespace:
                #                         doc_tokens.append(c)
                #                     else:
                #                         doc_tokens[-1] += c
                #                     prev_is_whitespace = False # 中文不需要
                doc_tokens.append(c)  # 每个token
                char_to_word_offset.append(len(doc_tokens) - 1)  # 每个字的index

            for qa in paragraph['qas']:
                qas_id = qa['id']
                question_text = qa['question']
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if is_training:
                    if version_2_with_negative:
                        is_impossible = qa['is_impossible']
                    if len(qa['answers']) == 0:

                        continue
                    if (len(qa['answers']) != 1) and (not is_impossible):  # 有多个答案的情况
                        raise ValueError(
                            "For training, each question should have exactly 1 answer."
                        )
                    if not is_impossible:
                        answer = qa['answers'][0]
                        orig_answer_text = answer['text']
                        answer_offset = answer['answer_start']
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        actual_text = "".join(doc_tokens[start_position:(end_position + 1)])

                        cleaned_answer_text = " ".join(tokenization_bert.whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            print("Could not find answer: '%s' vs '%s'", actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                )
                # print(example)
                examples.append(example)
    return examples


def convert_examples_to_features(
        examples,
        tokenizer,
        max_query_length,
        is_training,
        max_seq_length,
        doc_stride
):
    """
    具体如何将文本数据转成特征值
    :param examples: 第1步(read_squad_examples)读取进来的examples
    :param tokenizer: 模型对应的tokenizer
    :param max_query_length:需要定义的最长问题（query）长度，若query超过该长度，则会截断只取前半部分
    :param is_training:是否为训练数据集，若是将会返回开始结束具体位置，否则将该部分置为None
    :param max_seq_length:需要定义的最长长度（BERT常用512），包括'[CLS]'+query+'[SEP]'+answer+'[SEP]'，即query和answer的长度是max_seq_length-3
    :param doc_stride:滑窗法切分超长文档中用到的步长（下面会细说）。（这个值不要设置的太小！！因为是取min，会挤爆你的内存！！血的教训）
    :return:每一个example所转换成的feature的InputFeatures实例组成的list
    """
    """问题若超过max_query_length则会截断取前半部分，
    文档若超过max_seq_length则会使用滑窗法"""
    unique_id = 1000000000

    feature = []

    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        # 下面这段主要针对英文，有前缀、后缀，中文则会去掉空格
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)

            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1

        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3  # 3:[CLS],[SEP],[SEP]

        # 滑窗法
        _DocSpan = collections.namedtuple('DocSpan', ['start', 'length'])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)
            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(0)
            tokens.append("[SEP]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]
                is_max_context = _check_is_max_context(doc_spans, doc_span_index, split_token_index)

                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(1)
            tokens.append("[SEP]")
            segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # zero-pad up to the sequence length
            input_mask = [1] * len(input_ids)

            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            start_position = None
            end_position = None
            if is_training and not example.is_impossible:
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                # query是否在doc_span中
                if not (tok_start_position >= doc_start and tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = max_seq_length
                    end_position = max_seq_length
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0

            if example_index < 3:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()
                ]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    logger.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = "".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info("answer: %s" % (answer_text))

            feature.append(InputFeatures(unique_id=unique_id,
                                         example_index=example_index,
                                         doc_span_index=doc_span_index,
                                         tokens=tokens,
                                         token_to_orig_map=token_to_orig_map,
                                         token_is_max_context=token_is_max_context,
                                         input_ids=input_ids,
                                         input_mask=input_mask,
                                         segment_ids=segment_ids,
                                         start_position=start_position,
                                         end_position=end_position,
                                         is_impossible=example.is_impossible))
            unique_id += 1
    return feature


class SquadExample(object):
    """example的格式定义"""

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=False
                 ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):  # 显示属性
        s = ""
        s += "qas_id: %s" % self.qas_id
        s += ", question_text: %s" % self.question_text
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % self.start_position
        if self.end_position:
            s += ", end_position: %d" % self.end_position
        if self.is_impossible:
            s += ", is_impossible: %r" % self.is_impossible
        return s


class InputFeatures(object):
    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None
                 ):
        """
        将数据转换为特征值
        :param unique_id:自定义的一个id值
        :param example_index:第几个example
        :param doc_span_index:每个example中的第几个context（某些数据集会有多个）
        :param tokens:每个文本内容转化成的tokens的数组，包含"[CLS]","[SEP]","[SEP]"，但是原文中若含有空格则不会算成一个token
        :param token_to_orig_map:token映射回原文中的字词（英文中比较有用，因为中文基本没有做样式转换）
        :param token_is_max_context:该token是否在上下文中的位置最佳（下面会细说）
        :param input_ids:tokenizer将tokens映射成的ids数组
        :param input_mask:由0，1组成的数组，为了标识是否为补位（若长度不够长则补0，这时input_mask为0；其他为1）
        :param segment_ids:由0，1组成的数组，为了标识是否为同一个segment（比如query的segment_ids为0，answer的segment_ids为1，若还有下一句话则为2 ... ）
        :param start_position:与answers中的answer_start关联
        :param end_position:利用answers中text的长度和answer_start计算得出
        :param is_impossible:上面的数据中没有，表示是否拒识，即该问题是否有答案（squad 2.0增加了这类拒识问题）
        """
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    与我们将从哪个模型/config/tokenizer进行微调有关的参数。
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    与我们将输入模型用于训练和评估的数据有关的参数。
    """

    data_dir: Optional[str] = field(
        default=None, metadata={"help": "The input data dir. Should contain the .json files for the SQuAD task."}
    )
    use_tfds: Optional[bool] = field(default=True, metadata={"help": "If TFDS should be used or not."})
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    max_query_length: int = field(
        default=64,
        metadata={
            "help": "The maximum number of tokens for the question. Questions longer than this will "
                    "be truncated to this length."
        },
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
                    "and end predictions are not conditioned on one another."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, the SQuAD examples contain some that do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0, metadata={"help": "If null_score - best_non_null is greater than the threshold predict null."}
    )
    n_best_size: int = field(
        default=20, metadata={"help": "If null_score - best_non_null is greater than the threshold predict null."}
    )
    lang_id: int = field(
        default=0,
        metadata={
            "help": "language id of input for language-specific xlm models (see tokenization_xlm.PRETRAINED_INIT_CONFIGURATION)"
        },
    )


def load_and_cache_examples(args, tokenizer, evaluate=False, output_examples=False):
    """

    :param args:命令行输入的入参，具体如下
    :param tokenizer: 模型对应的tokenizer
    :param evaluate:是否要进行验证
    :param output_examples: 是否返回examples和features，training的时候为False，evalute的时候为True
    :return:
            output_examples==True时：dataset：5个纬度(all_input_ids, all_input_masks, all_segment_ids, all_start_positions, all_end_positions)的TensorDataset； examples：上面的read_squad_examples函数的输出；features：上面的convert_examples_to_features函数的输出
            output_examples==False时：dataset：5个纬度(all_input_ids, all_input_masks, all_segment_ids, all_start_positions, all_end_positions)的TensorDataset
    """
    if args.local_rank not in [-1, 0] and not evaluate:
        # 同步所有的进程, 直到整组(也就是所有节点的所有GPU)到达这个函数的时候, 才会执行后面的代码
        # Make sure only the first process in distributed training process the dataset, and the others will use the cache
        torch.distributed.barrier()

    # load data features from cache or dataset file
    # 从缓存或数据集文件加载数据功能
    input_dir = args.data_dir if args.data_dir else "."
    cached_features_file = os.path.join(input_dir,
                                        "cached_{}_{}_{}.pth".format(
                                            'dev' if evaluate else 'train',
                                            list(filter(None, args.model_name_or_path.split("/"))).pop(),
                                            # filter(None, ):把序列中的False值，如空字符串、False、[]、None、{}、()等等都丢弃
                                            str(args.max_seq_length)
                                        ))
    print("cached_features_file", cached_features_file)
    # Init features and dataset from cache if it exists
    # 从缓存中初始化功能和数据集（如果存在）
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        logger.info("Loading features from cached file %s", cached_features_file)
        features_and_dataset = torch.load(cached_features_file)
        features, dataset, examples = (
            features_and_dataset['features'],
            features_and_dataset['dataset'],
            features_and_dataset['examples']
        )
    else:
        logger.info("Creating features from dataset file at %s", input_dir)
        if not args.data_dir and ((evaluate and not args.dev_file) or (not evaluate and not args.train_file)):
            try:
                import tensorflow_datasets as tfds
            except ImportError:
                raise ImportError("If not data_dir is specified, tensorflow_datasets needs to be installed.")

            if args.version_2_with_negative:
                logger.warn("tensorflow_datasets does not handle version 2 of SQUAD.")

            tfds_examples = tfds.load("squad")
            examples = SquadV1Processor().get_examples_from_dataset(tfds_examples, evaluate=evaluate)
        else:
            if evaluate:

                input_file = os.path.join(args.data_dir, args.dev_file)
            else:
                input_file = os.path.join(args.data_dir, args.train_file)
            examples = read_squad_examples(input_file, is_training=not evaluate,
                                           version_2_with_negative=args.version_2_with_negative)

        features = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_query_length=args.max_query_length,
            is_training=not evaluate,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride
        )

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_input_masks = torch.tensor([f.input_mask for f in features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)

        if not evaluate:
            all_start_positions = torch.tensor([f.start_position for f in features], dtype=torch.long)
            all_end_positions = torch.tensor([f.end_position for f in features], dtype=torch.long)

            dataset = TensorDataset(
                all_input_ids,
                all_input_masks,
                all_segment_ids,
                all_start_positions,
                all_end_positions
            )
        else:
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)  # 每个词的位置
            dataset = TensorDataset(
                all_input_ids,
                all_input_masks,
                all_segment_ids,
                all_example_index
            )

        if args.local_rank in [-1, 0]:
            logger.info("Saving features into cached file %s", cached_features_file)
            torch.save({'features': features, 'dataset': dataset, 'examples': examples}, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()

    if output_examples:
        return dataset, examples, features
    return dataset

def open_write(data):
    with open("datas.json", "a+", encoding="utf-8") as a:
        a.write(str(data))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--local_rank', type=int, help='分布式计算用到的进程编号，-1表示不使用分布式')
    parser.add_argument('--data_dir', type=str, help='数据文件目录地址')
    parser.add_argument('--train_file', type=str, help='训练数据文件名')
    parser.add_argument('--dev_file', type=str, help='验证数据文件名')
    parser.add_argument('--model_name_or_path', type=str, help='模型名称或者路径')
    parser.add_argument('--max_query_length', type=int, help='最长问题长度')
    parser.add_argument('--max_seq_length', type=int, help='最长文本段落长度')
    parser.add_argument('--overwrite_cache', type=bool, help='保存的cache是否覆盖')
    parser.add_argument('--version_2_with_negative', type=bool, help='是否拒识')
    parser.add_argument('--doc_stride', type=int, help='滑窗法步长')
    args = parser.parse_args()




    # parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TFTrainingArguments))
    # print("获取模型1", parser)
    # model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # print("获取模型2")
    # print("model_args.tokenizer_name", model_args.tokenizer_name)
    # print("model_args.tokenizer_name", model_args.tokenizer_name)
    # print("model_args.model_name_or_path", model_args.model_name_or_path)
    # print("cache_dir=model_args.cache_dir", model_args.cache_dir)
    # print("use_fast=model_args.use_fast", model_args.use_fast)
    tokenizer = BertTokenizer.from_pretrained(
        None if None else "bert-base-chinese",
        cache_dir=None,
        use_fast=False,
    )
    print("获取了模型对应的tokenizer", tokenizer)
    # print(data_args.doc_stride)
    # feature = convert_examples_to_features(examples=example,
    #                              tokenizer=tokenizer,
    #                              max_query_length=data_args.max_query_length,
    #                              is_training=True,
    #                              max_seq_length=data_args.max_seq_length,
    #                              doc_stride=data_args.doc_stride)
    # print(feature)
    # args["max_query_length"] = data_args.max_query_length
    # args["max_seq_length"] = data_args.max_seq_length
    # args["overwrite_cache"] = False
    # args["version_2_with_negative"] = False
    # args["doc_stride"] = data_args.doc_stride
    # print(args)
    # cached_features_file = "D:\\MRC\\squad-zen\\cached_train_bert-base-uncased_128.pth"
    # torch.save({'features': "features", 'dataset': "dataset", 'examples': "examples"}, cached_features_file)

    dataset = load_and_cache_examples(args, tokenizer=tokenizer)
    print(dataset)
    print("训练完成")
    # open_write(dataset)
    # print("存储完成")












