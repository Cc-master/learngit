# -*- coding: utf-8 -*-
# @Author : wgq
# @time   : 2023/1/5
# @File   : train_mrc_gg.py
# Software: PyCharm
# explain: 读取处理完成的语料  开始训练模型

import argparse
import distutils.util

import collections
import json
import os
import string
import zhon.hanzi
import sys
import timeit
from log import logger
from typing import Optional
import hashlib
from dataclasses import dataclass, field

import torch
import torch.distributed 
from torch.optim import AdamW
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from torch.utils.tensorboard import SummaryWriter
# from torchaudio.utils.sox_utils import set_seed

# from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer, AutoModelForMaskedLM, set_seed, BertForQuestionAnswering
# from transformers import set_seed
from tqdm import tqdm, trange
from transformers.data.metrics.squad_metrics import compute_predictions_log_probs, compute_predictions_logits
from transformers.data.processors.squad import SquadV1Processor, TensorDataset
from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    TFAutoModelForQuestionAnswering,
    TFTrainer,
    TFTrainingArguments,
    squad_convert_examples_to_features,
    get_linear_schedule_with_warmup)
from transformers.models.bert import tokenization_bert
sys.path.append("D:\demo_kafka")
from mrc.json_open import load_and_cache_examples


class SquadResult(object):
    def __init__(self, unique_id, start_logits, end_logits, start_top_index=None, end_top_index=None, cls_logtis=None):
        """
        结果格式定义
        :param unique_id: 自定义的一个id值
        :param start_logits: softmax之前的每一位的为开始的概率
        :param end_logits: softmax之前的每一位的为结束的概率
        :param start_top_index: （XLNET、XLM独有）：默认为None，前n个开始概率大的indices
        :param end_top_index:（XLNET、XLM独有）：默认为None，前n个结束概率大的indices
        :param cls_logtis:（XLNET、XLM独有）：默认为None，当有拒识问题时，CLS标签表示了拒识概率
        """
        self.start_logits = start_logits
        self.end_logits = end_logits
        self.unique_id = unique_id
        if start_top_index:
            self.start_top_index = start_top_index
            self.end_top_index = end_top_index
            self.cls_logits = cls_logtis


class GoldAnswer(object):
    def __init__(self,
                 qas_id,
                 answers
                 ):
        """
        正确答案格式定义
        :param qas_id:每组问题、文档、答案的id
        :param answers:正确答案
        """
        self.qas_id = qas_id
        self.answers = answers




def get_tokens(s):  #（不套娃了，写在一起...）这个函数是为了把答案字符串变成单个token组成的list
    if not s:
        return []
    return normalize_answer(s).split()
def normalize_answer(s):
    """
    答案标准化
    :param s: 字符串
    :return:white_space_fix(remove_punc(lower(s))): 经过英文转小写、去除标点符号、增加空格来让答案和预测结果格式一致
    """
    # def remove_articles(text): # 待补充
    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation + zhon.hanzi.punctuation)  # 英文标点+中文标点
        return "".join(ch for ch in text if ch not in exclude)
        # return "".join(ch for ch in text if ch.isdigit() or ch.isalpha()) # 2019CAIl方法

    def lower(text):
        return text.lower()

    return white_space_fix(remove_punc(lower(s)))
def compute_exact(a_gold, a_pred):
    """
    计算绝对相同的分数
    :param a_gold:一个标准答案字符串
    :param a_pred:一个预测答案字符串
    :return:int(normalize_answer(a_gold)== normalize_answer(a_pred))：bool，预测答案是否和标准答案完全一致
    """
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))
def compute_f1(a_gold, a_pred):
    """
    计算f1分数
    :param a_gold:一个标准答案字符串
    :param a_pred:一个预测答案字符串
    :return:f1: 由每个字计算的f1值
    """
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)  # 字典取交集
    num_same = sum(common.values()) # 相同的字的个数
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
def get_raw_scores(examples, preds):
    """
    计算原始分数
    :param examples:同上 squad_evaluate方法
    :param preds:同上 squad_evaluate方法
    :return:exact_scores: {'id': exact_score}
            f1_scores: {'id': f1_score}
    """
    exact_scores = {}
    f1_scores = {}

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer for answer in example.answers if normalize_answer(answer)]

        if not gold_answers:
            gold_answers = [""]  # 无法回答问题

        if qas_id not in preds:
            print("Missing prediction for %s" % qas_id)

        prediction = preds[qas_id]
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers) # 可能有多个标准答案
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)

    return exact_scores, f1_scores

def gold_answers(data_path):
    """
    验证集正确答案的提取
    :param data_path:验证数据集的路径
    :return:GoldAnswer实例组成的list
    """
    print(data_path)
    dataset = json.load(open(data_path))
    gold_list = []
    for para in dataset['data'][0]['paragraphs']:
        qas = para['qas']
        for qa in qas:
            qid = qa['id']
            gold_answers = []
            if not qa['answers']:
                gold_answers = [''] # 若无答案，则为拒答，''
            for answer in qa['answers']:  # 多个答案
                gold_answers.append(answer['text'])
            if qid in [example.qas_id for example in gold_list]:
                sys.stderr.write("Gold file has duplicate ids: {}".format(qid))
            gold_list.append(GoldAnswer(
                qas_id=qid,
                answers=gold_answers
            ))
    return gold_list

def apply_no_ans_threshold(scores, na_probs, qid_to_has_ans, na_prob_thresh):
    """
    无答案阈值 （bert论文：这个阈值需要根据dev数据集来选择，为了能达到最大F1值）
    :param scores:分数（exact分数或f1分数）
    :param na_probs:{'id': 无答案概率}
    :param qid_to_has_ans:{qid: True/False} qid有无答案的字典
    :param na_prob_thresh:无答案阈值（默认为1.0）
    :return:new_scores：经过修改的新分数，如果id的无答案概率大于阈值，那么预测有答案的分数为0.0，预测为无答案的分数为1.0。
    """
    new_scores = {}
    for qid, s in scores.items():
        pred_na = na_probs[qid] > na_prob_thresh
        if pred_na:
            new_scores[qid] = float(not qid_to_has_ans[qid])
        else:
            new_scores[qid] = s
    return new_scores

def make_eval_dict(exact_scores, f1_scores, qid_list=None):
    """
    汇总字典
    :param exact_scores:exact分数
    :param f1_scores:f1分数
    :param qid_list:qid列表
    :return:{'exact': exact总平均分, 'f1': f1总平均分, 'total': qid总个数}
    """
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ('exact', 100.0 * sum(exact_scores.values()) / total),
                ('f1', 100.0 * sum(f1_scores.values()) / total),
                ('total', total)
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ('exact', 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ('f1', 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ('total', total)
            ]
        )

def merge_eval(main_eval, new_eval, prefix):
    """
    # 在总的汇总字典上添加明细字典
    :param main_eval:
    :param new_eval:
    :param prefix:
    :return:
    """
    for k in new_eval:
        main_eval["%s_%s" % (prefix, k)] = new_eval[k]

def find_best_thresh(preds, scores, na_probs, qid_to_has_ans):
    """
    找到最佳无答案阈值
    :param preds:同上
    :param scores:分数（exact分数或f1分数）
    :param na_probs:同上
    :param qid_to_has_ans:同上
    :return:best_thresh: 能得到最高分的无答案阈值
    """
    num_no_ans = sum(1 for k in qid_to_has_ans if not qid_to_has_ans[k]) # 无答案个数
    cur_score = num_no_ans
    best_score = cur_score
    best_thresh = 0.0
    qid_list = sorted(na_probs, key=lambda k: na_probs[k])
    for _, qid in enumerate(qid_list):
        if qid not in scores:
            continue
        if qid_to_has_ans[qid]:
            diff = scores[qid]
        else:
            if preds[qid]:
                diff = -1
            else:
                diff = 0
        cur_score += diff
        if cur_score > best_score:
            best_score = cur_score
            best_thresh = na_probs[qid]
    return 100.0 * best_score / len(scores), best_thresh

def find_all_best_thresh(main_eval, preds, exact_raw, f1_raw, na_probs, qid_to_has_ans):
    """
    找到所有最佳无答案阈值
    :param main_eval:3.3中汇总的字典
    :param preds:同上
    :param exact_raw:未经过无答案阈值的exact分数
    :param f1_raw:未经过无答案阈值的f1分数
    :param na_probs:同3.2
    :param qid_to_has_ans:同3.2
    :return:main_eval中新增'best_exact', 'best_exact_thresh', 'best_f1', 'best_f1_thresh'
    """
    best_exact, exact_thresh = find_best_thresh(preds, exact_raw, na_probs, qid_to_has_ans)
    best_f1, f1_thresh = find_best_thresh(preds, f1_raw, na_probs, qid_to_has_ans)

    main_eval['best_exact'] = best_exact
    main_eval['best_exact_thresh'] = exact_thresh
    main_eval['best_f1'] = best_f1
    main_eval['best_f1_thresh'] = f1_thresh

def squad_evaluate(examples, preds, no_answer_probs=None, no_answer_probability_threshold=1.0):
    """
    squad_evaluate
    :param examples: examples数据
    :param preds:由1.预测出的预测数据({'id': 'text'})
    :param no_answer_probs:{'id': 无答案概率}
    :param no_answer_probability_threshold:无答案阈值（默认为1.0）
    :return: {'exact': exact总平均分, 'f1': f1总平均分, 'total': qid总个数, ('exact_HasAns':, 'f1_HasAns':, 'total_HasAns':, 'exact_NoAns':, 'f1_NoAns':, best_exact':, 'best_exact_thresh':, 'best_f1':, 'best_f1_thresh':)} （括号内为可选项）
    """
    qas_id_to_has_answer = {example.qas_id: bool(example.answers) for example in examples}
    has_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if has_answer]
    no_answer_qids = [qas_id for qas_id, has_answer in qas_id_to_has_answer.items() if not has_answer]

    if no_answer_probs is None:
        no_answer_probs = {k: 0.0 for k in preds}

    exact, f1 = get_raw_scores(examples, preds)

    exact_threshold = apply_no_ans_threshold(
        exact, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold
    )
    f1_threshold = apply_no_ans_threshold(f1, no_answer_probs, qas_id_to_has_answer, no_answer_probability_threshold)

    evaluation = make_eval_dict(exact_threshold, f1_threshold)

    if has_answer_qids: # 有答案的汇总一版
        has_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=has_answer_qids)
        merge_eval(evaluation, has_ans_eval, "HasAns")

    if no_answer_qids: # 无答案的汇总一版
        no_ans_eval = make_eval_dict(exact_threshold, f1_threshold, qid_list=no_answer_qids)
        merge_eval(evaluation, no_ans_eval, "NoAns")

    if no_answer_probs:
        find_all_best_thresh(evaluation, preds, exact, f1, no_answer_probs, qas_id_to_has_answer)

    return evaluation




def evaluate(args, model, tokenizer, prefix="text"):
    """

    :param args:入参：
    :param model:模型
    :param tokenizer:分词器
    :param prefix:前缀（其实这里准确地说是后缀），保存的时候加在名字后半部分便于区分。
    :return:
    """
    if args.do_predict:
        # dataset, examples, features = load_and_cache_examples(args, tokenizer, predict=True, output_examples=True)
        dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)
    else:
        dataset, examples, features = load_and_cache_examples(args, tokenizer, evaluate=True, output_examples=True)

    if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(args.output_dir)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(dataset) # 这里按顺序取，区别于train的随机
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
        model = torch.nn.DataParallel(model)

    # Start Eval
    logger.info("***** Running evaluation {} *****".format(prefix))
    logger.info(" Num examples = %d", len(dataset))
    logger.info(" Batch size = %d", args.eval_batch_size)

    all_results = []
    start_time = timeit.default_timer() # 计时

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)

        with torch.no_grad():
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2]
            }
            if args.model_type in ['xlm', 'roberta', 'distilbert', 'camembert']:
                del inputs['token_type_ids']

            example_indices = batch[3]
            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[4], 'p_mask': batch[5]})
                if hasattr(model, 'config') and hasattr(model.config, 'lang2id'):
                    inputs.update(
                        {'langs': (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)})

            outputs = model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            # print("outputs", "outputs")
            # for outputsss in outputs:
            #     print(outputs[outputsss][i])
            # print("outputs", "outputs")
            output = [(outputs[output][i]).detach().cpu().tolist() for output in outputs]
            if len(output) >= 5: # ['xlnet', 'xlm']的返回多包含几个参数
                start_logits = output[0]
                start_top_index = output[1]
                end_logits = output[2]
                end_top_index = output[3]
                cls_logits = output[4]

                result = SquadResult(unique_id, start_logits, end_logits, start_top_index, end_top_index, cls_logits)
            else:
                start_logits, end_logits = output
                result = SquadResult(unique_id, start_logits, end_logits)

            all_results.append(result)

    evalTime = timeit.default_timer() - start_time # 计算eval过程时长
    logger.info(" Evaluation done in total %f secs (%f sec per example)", evalTime, evalTime / len(dataset))

    output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
    output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))

    if args.version_2_with_negative: # 该文件记录的是每个问题拒识的可能性，如果该值大于某个阀值(可调整)，则该问题无法回答
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds_{}.json".format(prefix))
    else:
        output_null_log_odds_file = None

    if args.model_type in ['xlnet', 'xlm']:
        start_n_top = model.config.start_n_top if hasattr(model, 'config') else model.module.config.start_n_top
        end_n_top = model.config.end_n_top if hasattr(model, 'config') else model.module.config.end_n_top

        predictions = compute_predictions_log_probs(
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            start_n_top,
            end_n_top,
            args.version_2_with_negative,
            tokenizer,
            args.verbose_logging
        )
    else:
        predictions = compute_predictions_logits( # 见下方1.
            examples,
            features,
            all_results,
            args.n_best_size,
            args.max_answer_length,
            args.do_lower_case,
            output_prediction_file,
            output_nbest_file,
            output_null_log_odds_file,
            args.verbose_logging,
            args.version_2_with_negative,
            args.null_score_diff_threshold,
            tokenizer
        )
    print("args.evaluate_during_training", args.evaluate_during_training)
    print("args.do_predict", args.do_predict)
    # compute F1 and exact scores.
    if args.evaluate_during_training and not args.do_predict:
        examples = gold_answers(os.path.join(args.data_dir, args.dev_file)) # 见下方2.
        results = squad_evaluate(examples, predictions) # 见下方3.
        return results




def train(train_dataset, model, args):
    """Train the model 训练模型
    train_dataset: 训练数据集，由《机器阅读理解run_squad源码研读（上）》生成的dataset实例
    model: 模型
    args: 命令行输入的入参，具体如下
    :return  global_step：全部step数, tr_loss / globalstep：全局平均loss值，其中tr_loss是所有epoch的loss的累加
    """


    if args.local_rank in [-1, 0]:
        tb_writer = SummaryWriter() # 写tensorboard文件

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    # 数据部分
    train_sampler = RandomSampler(train_dataset)  # shuffle indices
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # 计算总共跑了多少个steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]  # model.named_parameters() 打印模型的每个参数的名字和参数值
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=t_total)  # 学习率在warmup（线性增加）之后进行线性衰减

    # Check if saved optimizer or scheduler states exist
    if os.path.isfile(os.path.join(args.model_name_or_path, 'optimizer.pt')) and os.path.isfile(
            os.path.join(args.model_name_or_path, 'scheduler.pt')):
        optimizer.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'optimizer.pt')))
        scheduler.load_state_dict(torch.load(os.path.join(args.model_name_or_path, 'scheduler.pt')))

    # 混合精度训练
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError('Please install apex!')

        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
        model.to(torch.device(args.device))
    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank, find_unused_parametes=True)

    # Start Train
    logger.info("****** Running training ******")
    logger.info(" Num examples = %d", len(train_dataset))
    logger.info(" Num Epochs = %d", args.num_train_epochs)
    logger.info(" Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(" Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info(" Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info(" Total optimization steps = %d", t_total)

    global_step = 1
    epochs_trained = 0
    steps_trained_in_current_epoch = 0

    # Check if continuing training from a checkpoint
    if os.path.exists(args.model_name_or_path):
        try:
            checkpoint_suffix = args.model_name_or_path.split('-')[-1].split('/')[
                0]  # set global_step to global_step of last saved checkpoint from model path
            global_step = int(checkpoint_suffix)
            epochs_trained = global_step // (len(train_dataloader) // args.gradient_accumulation_steps)
            steps_trained_in_current_epoch = global_step % (len(train_dataloader // args.gradient_accumulation_steps))

            logger.info(" Continuing training from checkpoint, will skip to saved global_step")
            logger.info(" Continuing training from epoch %d", epochs_trained)
            logger.info(" Continuing training from global step %d", global_step)
            logger.info(" Will skip the first %d steps in the first epoch", steps_trained_in_current_epoch)

        except ValueError:
            logger.info(" Starting fine-tuning.")

    tr_loss, logging_loss = 0.0, 0.0

    # 请记住Pytorch会累加梯度, 每次训练前需要清空梯度值
    model.zero_grad()
    train_iterator = trange(epochs_trained, int(args.num_train_epochs), desc='Epoch',
                            disable=args.local_rank not in [-1, 0])

    set_seed(args.seed) # 源代码中设置了2次随机种子，这里我不太明白，请明白的同学教教我！

    for _ in train_iterator: # 每个epoch
        epoch_iterator = tqdm(train_dataloader, desc='Iteration', disable=args.local_rank not in [-1, 0])

        for step, batch in enumerate(epoch_iterator): # 每个batch
            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)  # 因为tuple不可改变且效率高，故把list转成tuple
            # dataloader中的数据，对应上一篇文章中load_and_cache_examples中返回的dataset
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_type_ids': batch[2],
                'start_positions': batch[3],
                'end_positions': batch[4]
            }

            if args.model_type in ['xlm', 'roberta', 'distilbert', 'camembert']:
                del inputs['token_type_ids']  # 当有两句话输入的时候才有，上述模型没有NSP

            if args.model_type in ['xlnet', 'xlm']:
                inputs.update({'cls_index': batch[5], 'p_mask': batch[6]})
                if args.version_2_with_negative:  # 是否有拒识问题
                    inputs.update({'is_impossible': batch[7]})
                if hasattr(model, 'config') and hasattr(model.config, 'lang2id'):  # 检查一个对象是否含有属性
                    inputs.update(
                        {'langs': (torch.ones(batch[0].shape, dtype=torch.int64) * args.lang_id).to(args.device)})

            outputs = model(**inputs)
            loss = outputs[0]

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel (not distributed) training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()  # 损失放大（Loss Scaling）即使用了混合精度训练，还是会存在无法收敛的情况，原因是激活梯度的值太小，造成了下溢出（Underflow
                    # ）。反向传播前，将损失变化（dLoss）手动增大倍，因此反向传播时得到的中间变量（激活函数梯度）则不会溢出；反向传播后，将权重梯度缩小倍，恢复正常值。
            else:
                loss.backward()

            tr_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:  # 每次梯度累加全部完成，做一个梯度剪切
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()  # 更新参数
                scheduler.step()  # 更新学习率
                model.zero_grad()
                global_step += 1

                # tensorboard
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    # Only evaluate when single GPU otherwise metrics may not average well
                    if args.local_rank == -1 and args.evaluate_during_training:
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            tb_writer.add_scalar('eval_{}'.format(key), value, global_step)
                    tb_writer.add_scalar('lr', scheduler.get_lr()[0], global_step)
                    tb_writer.add_scalar('loss', (tr_loss - logging_loss) / args.logging_steps, global_step)
                    logging_loss = tr_loss

                # save model checkpoint

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Take care of distributed/parallel training
                    model_to_save = model.module if hasattr(model, 'module') else model
                    model_to_save.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info('Saving model checkpoint to %s', output_dir)

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
                    logger.info('Saving optimizer and scheduler states to %s', output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()

        if 0 < args.max_steps < global_step:
            train_iterator.close()

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, tr_loss / global_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='manual to this script')

    # parser.add_argument('--local_rank', type=int, help='分布式计算用到的进程编号，-1表示不使用分布式')
    parser.add_argument('--data_dir', type=str, help='数据文件目录地址')
    parser.add_argument('--train_file', type=str, help='训练数据文件名')
    parser.add_argument('--dev_file', type=str, help='验证数据文件名')
    # parser.add_argument('--model_name_or_path', type=str, help='模型名称或者路径')
    parser.add_argument('--max_query_length', type=int, help='最长问题长度')
    parser.add_argument('--max_seq_length', type=int, help='最长文本段落长度')
    parser.add_argument('--overwrite_cache', type=bool, help='保存的cache是否覆盖')
    # parser.add_argument('--version_2_with_negative', type=bool, help='是否拒识')
    parser.add_argument('--doc_stride', type=int, help='滑窗法步长')

    parser.add_argument('--max_steps', type=int, help='最多运行多少步，若此处设置>0，将会覆盖参数num_train_epochs')
    parser.add_argument('--num_train_epochs', type=int, help='训练的epoch数')
    parser.add_argument('--gradient_accumulation_steps', type=int, help='梯度累加的步长大小，即轮数')
    parser.add_argument('--weight_decay', type=float, help='权重衰减')
    parser.add_argument('--learning_rate', type=float, help='学习率')
    parser.add_argument('--adam_epsilon', type=float, help='给Adam优化器定义epsilon值，防止除0错误')
    parser.add_argument('--warmup_steps', type=int, help='线性warmup的轮数')
    parser.add_argument('--model_name_or_path', type=str, help='模型所在目录（若使用的是transformers则非必须）')
    parser.add_argument('--fp16', type=bool, help='是否使用fp16混合精度')
    parser.add_argument('--fp16_opt_level', type=str, help="fp16的等级，['O0', 'O1', 'O2', 'O3']，具体见amp文档")
    parser.add_argument('--n_gpu', type=int, help="使用几个gpu")
    parser.add_argument('--local_rank', type=int, help="分布式计算用到的进程编号，-1表示不使用分布式")
    parser.add_argument('--per_gpu_train_batch_size', type=int, help="每个gpu的训练batch大小")
    parser.add_argument('--seed', type=int, help="随机种子")
    parser.add_argument('--model_type', type=str, help="('distilbert', 'albert', 'roberta', 'bert', 'xlnet', 'flaubert', 'xlm')")
    parser.add_argument('--version_2_with_negative', type=bool, help="是否拒识")
    parser.add_argument('--lang_id', type=int, help="针对语言有要求的xlm模型的语言id标识")
    parser.add_argument('--max_grad_norm', type=float, help="梯度裁剪的max norm值，防止梯度爆炸")
    parser.add_argument('--logging_steps', type=int, help="打log的步长")
    parser.add_argument('--evaluate_during_training', type=lambda x:bool(distutils.util.strtobool(x)), help="是否在训练的时候做evaluate")
    parser.add_argument('--save_steps', type=int, help="保存模型及其参数的步长")
    parser.add_argument('--output_dir', type=str, help="输出目录路径")
    parser.add_argument('--device', type=str, help="指定gpu")
    parser.add_argument('--do_predict', type=lambda x:bool(distutils.util.strtobool(x)), help="是否做预测")
    parser.add_argument('--per_gpu_eval_batch_size', type=int, help="每gpu评估批大小")
    parser.add_argument('--n_best_size', type=int, help="每个问题推出n个最优答案")
    parser.add_argument('--max_answer_length', type=int, help="设置的最长答案字数（若答案字数超出，会被丢弃）")
    parser.add_argument('--verbose_logging', type=int, help="log显示级别")
    parser.add_argument('--do_lower_case', type=bool, help="是否大小写敏感")
    parser.add_argument('--null_score_diff_threshold', type=float, help="拒答阈值（拒答：score_null - 第一名有答案的(start_logit+end_logit) > 阈值，详见下述）")

    args = parser.parse_args()

    # tokenizer = AutoTokenizer.from_pretrained(
    #     None if None else "bert-base-chinese",
    #     cache_dir=None,
    #     use_fast=False,
    # )
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    device = 'cuda:0'
    model_name = 'bert-base-chinese'
    model_path = ''
    # model = BertModel.from_pretrained("bert-base-chinese").to(device)
    model = BertForQuestionAnswering.from_pretrained("bert-base-chinese").to(device)
    # print("获取模型", model)
    dataset = load_and_cache_examples(args, tokenizer=tokenizer)

    print("获取了模型对应的tokenizer", tokenizer)
    # train_dataset = "D:\MRC\squad-zen\cached_train_bert-base-uncased_128.pth"
    print("开始训练")
    global_step, globa = train(train_dataset=dataset, model=model, args=args)
    print("训练结果")
    print("global_step：{}".format(global_step))
    print("globa：{}".format(globa))
    print("训练结束")