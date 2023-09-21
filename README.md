# A Survey of Attributions for Large Language Models
### 出发点
自从基于大模型的开放域对话系统发布以来，这些系统一直面临着其生成内容存在不真实、不准确的问题。这类问题常被学术界称为幻象“Hallucination”问题，即生成的内容中存在编造模糊的事实。当面向信息查询和知识问答时，即人们依赖大模型提供专业知识的场景下，幻象问题变得突出。进一步，幻象大致分为两种，一种是内部幻象，指的是输入文本来源固定，而输出生成的文本与来源矛盾，比如在文本摘要和翻译任务中，摘要或翻译与文档内容不一致；一种是外部幻象，指的是输出生成文本无法被验证（可能为真或假，不能从输入文本来源推导出来），而且其生成的文本流畅内容丰富，使得人们很难单凭输入来源去验证真假。因此，相对于前者，外部幻象的重要度更高、解决的挑战更大，比如在现实世界的医疗专业领域，生成虚假内容的代价非常高。

幻象问题的本质是由于预训练的数据来源于现实世界的未加过滤的海量文本，而这些人类写的文本中本身就存在不实矛盾的内容，而预训练的目标仅仅是预测下一个词，并没有显式地对生成内容的真实性建模。即使利用了基于人类反馈的强化学习后，模型仍有可能存在外部幻象。因此为了解决外部幻象的问题，研究者也开始利用检索等措施来帮助对话系统更真实可靠。明确归因与强化学习的区别在于一方面为了人类验证合规的需要，另一方面则是生成内容随着时间变化有可能过时变成无效内容。

这些事实归因措施大致可以归纳为三种：（1）直接让大模型输出其回答的归因。但是这种方式往往会不仅答案存在幻象，就连输出的归因都存在幻象。有[研究](https://arxiv.org/pdf/2309.09401.pdf)表明ChatGPT在大约一半的情况下（约50.6%的时间内）提供了正确或部分正确的答案，但其建议的参考资料只有14%的时间存在;（2）检索后再回答。这种方法的出发点是通过明确的检索,然后让模型基于这些检索信息进行回答。但这种方式会存在输出内容中参数知识和外部检索知识边界不清楚、知识冲突的挑战。当然也可以把检索当做一种特殊的工具去让模型自己去触发，类似于GPT-4的citation插件;（3）生成后再归因。这种方法先进行回答问题，然后将问题加上答案一起进行检索以归因，然后再去修改答案并进行必要的归因。另外，现在的搜索引擎如Bing Chat已经支持了这种事实归因，但是有[研究](https://arxiv.org/pdf/2304.09848.pdf)表明只有51.5%来自四个生成式搜索引擎的生成内容完全得到了它们引用的参考文献的支持，而且这种归因在医学和法律等高风险专业领域远远达不到令人满意，[研究](https://arxiv.org/pdf/2309.07852.pdf)表明存在大量不完整的归因（分别有35%和31%的归因不完整），并且许多归因来自不可靠的来源（51%的归因被专家评定为不可靠）。挑战的本质在于归因具有两个要求：（1）应该全面地归因或者引用（高召回率；所有模型生成内容的每一个论述都得到完全支持的引用）；（2）应该准确地归因或者引用（高准确率，每一个引用都支持其相关的论述）。

相比于文本幻象的survey, 我们的出发点是更细化地偏向于事实归因的调研，包括当前事实归因的来源、事实归因的技术方法、事实归因的评价、事实归因的数据集，事实归因的有限性 （无法对模型参数知识进行归因、知识来源的正确与否：可能来自于未经验证的内容，也有可能来自于之前大模型生成的内容、过度归因导致信息过载和敏感信息传播风险、事实归因的偏见：过分喜欢某个来源或观点） （要补充）。本调研不关心内部幻象的检测与改进。另外，考虑到检索的来源可能本身是不正确的，相比于最终模型输出的事实正确性，本研究更关注于事实归因的忠实度。通过事实归因，模型的输出有了来源，减少了生成内容的不确定性，提升了生成内容的事实性、可验证性与可解释性，从而增加系统的可信性和可靠性。

归因也可能是迫不得已的中间手段，因为精确解释LLM行为的尝试都注定过于复杂，难以为任何人所理解，所以人们期望像搜索引擎的方式去对待大模型，一方面希望它能精确全面的告诉我们相关信息，另一方面又希望它告诉我们的信息有据可查。

## 1. Attribution Definition & Position Paper
*   [2021/07] **Rethinking Search: Making Domain Experts out of Dilettantes** *Donald Metzler et al. arXiv.* [[paper](https://arxiv.org/pdf/2105.02274.pdf)] 
      > This position paper says "For example, for question answering tasks our envisioned model is able to synthesize a singleanswer that incorporates information from many documents in the corpus, and it will be able to support assertions in the answer by referencing supporting evidence in the corpus, much like a properly crafted Wikipedia entry supports each assertion of fact with a link to a primary source. This is just one of many novel tasks that this type of model has the potential to enable."
*   [2023/03] **TRAK: Attributing Model Behavior at Scale** *Sung Min Park et al. arXiv.* [[paper](https://arxiv.org/abs/2303.14186)][[code](https://github.com/MadryLab/trak)]
      > Attributing Model: trace model predictions back to training data. This paper introduces a data attribution method that is both effective and computationally tractable for large-scale, differentiable models.

*   [2023/07] **Citation: A Key to Building Responsible and Accountable Large Language Models** *Jie Huang et al. arXiv.* [[paper](https://arxiv.org/pdf/2307.02185.pdf)] 
      > This position paper embarks on an exploratory journey into the potential of integrating a citation mechanism within large language models, examining its prospective benefits, the inherent technical obstacles, and foreseeable pitfalls.

*   [2023/09] **ChatGPT Hallucinates when Attributing Answers** *Guido Zuccon et al. arXiv.* [[paper](https://arxiv.org/abs/2309.09401)]
      > This paper suggests that ChatGPT provides correct or partially correct answers in about half of the cases (50.6% of the times), but its suggested references only exist 14% of the times. In thoses referenced answers, the reference often does not support the claims ChatGPT attributes to it.



## 2. Attribution Paper Before the Era of Large Language Models 







### 2.1 Fact Checking
*   [2021/08] **A Survey on Automated Fact-Checking** [[paper](https://arxiv.org/pdf/2108.11896.pdf)]

*   [2021/10] **Truthful AI: Developing and governing AI that does not lie** [[paper](http://arxiv.org/abs/2110.06674)]

### 2.2 Claim Verificication
*   [2021/05] **Evaluating Attribution in Dialogue Systems: The BEGIN Benchmark** *Nouha Dziri et al. TACL'22* [[paper](https://aclanthology.org/2022.tacl-1.62/)][[code](https://github.com/google/BEGIN-dataset)]

### 2.3 Feature Attribution and Interpretability of Models for NLP 
*   [2022/12] **Foveate, Attribute, and Rationalize: Towards Physically Safe and Trustworthy AI** *Alex Mei et al. findings of ACL'22* [[paper](https://aclanthology.org/2023.findings-acl.701.pdf)]

## 3. Sources of Attribution

### 3.1 Pre-training Data
* [2023/02] **The ROOTS Search Tool: Data Transparency for LLMs** *Aleksandra Piktus et al. arXiv.* [[paper](https://arxiv.org/pdf/2302.14035.pdf)]
* [2022/05] **ORCA: Interpreting Prompted Language Models via Locating Supporting Data Evidence in the Ocean of Pretraining Data** *Xiaochuang Han et al. arXiv.* [[paper](https://arxiv.org/pdf/2205.12600.pdf)]

* [2022/05] **Understanding In-Context Learning via Supportive Pretraining Data** *Xiaochuang Han et al. arXiv.* [[paper](https://arxiv.org/pdf/2306.15091.pdf)]
### 3.2 Out-of-model Knowledge


## 4. Datasets for Attribution
* [2022/12] **CiteBench: A benchmark for Scientific Citation Text Generation** *Martin Funkquist et al. arXiv.* [[paper](https://arxiv.org/abs/2212.09577)]


* [2023/05] **Enabling Large Language Models to Generate Text with Citations** *Tianyu Gao et al. arXiv.* [[paper](https://arxiv.org/pdf/2305.14627.pdf)] [[code](https://github.com/princeton-nlp/ALCE)]
   > This paper proposes ALCE dataset, which collects a diverse set of questions and retrieval corpora and requires building end-to-end systems to retrieve supporting evidence and generate answers with citations.

* [2023/07] **HAGRID: A Human-LLM Collaborative Dataset for Generative Information-Seeking with Attribution** *Ehsan Kamalloo et al. arXiv.* [[paper](https://arxiv.org/pdf/2307.16883.pdf)] [[code](https://github.com/project-miracl/hagrid)]
   > This paper introduces the HAGRID dataset for building end-to-end generative information-seeking models that are capable of retrieving candidate quotes and generating attributed explanations.

* [2023/09] **EXPERTQA : Expert-Curated Questions and Attributed Answers** *Chaitanya Malaviya et al. arXiv.* [[paper](https://arxiv.org/pdf/2309.07852.pdf)] [[code](https://github.com/chaitanyamalaviya/expertqa)]
   > This paper introduces the EXPERTQA, a high-quality long-form QA dataset with 2177 questions spanning 32 fields, along with verified answers and attributions for claims in the answers. 


## 5 Approaches to Attribution

### 5.1 Direct Generated Attribution

### 5.2 Retrieval-then-Answering

### 5.3 Post-Generation Attribution

*  [2022/10] **RARR: Researching and Revising What Language Models Say, Using Language Models** [[paper](https://arxiv.org/abs/2210.08726)]

*  [2023/04] **The Internal State of an LLM Knows When its Lying** [[paper](http://arxiv.org/abs/2304.13734)]
      > This paper utilizes the LLM's hidden layer activations to determine the veracity of statements by a classifier receiveing as input the activation values from the LLM for each of the statements in the dataset.

*  [2023/05] **Do Language Models Know When They're Hallucinating References?** [[paper](http://arxiv.org/abs/2305.18248)]
*  [2023/05] **Complex Claim Verification with Evidence Retrieved in the Wild** [[paper](https://arxiv.org/abs/2305.11859)][[code](https://github.com/jifan-chen/fact-checking-via-raw-evidence)]
      > This paper proposes a pipeline(claim decomposition, multi-granularity evidence retrieval, claim-focused summarization) to improve veracity judgments.
*  [2023/06] **Retrieving Supporting Evidence for LLMs Generated Answers** [[paper](http://arxiv.org/abs/2306.13781)]
      > This paper proposes a two-step verification. The LLM's answer and the retrieved document queried by question and LLM's answer are compared by LLM, checking whether the LLM's answer is hallucinated.



### 5.4 Attribution Systems & End-to-End Attribution Models

* [2022/03] **LaMDA: Language Models for Dialog Applications**  [[paper](https://arxiv.org/pdf/2201.08239.pdf)]

*  [2022/03] **WebGPT: Browser-assisted question-answering with human feedback**  [[paper](https://arxiv.org/pdf/2112.09332.pdf)]

*  [2022/03] **GopherCite - Teaching language models to support answers with verified quotes**  [[paper](http://arxiv.org/abs/2203.11147)]

## 6. Attribution Evaluation
*   [2021/12] **Measuring Attribution in Natural Language Generation Models.** *H Rashkin et al. CL.* [[paper](https://arxiv.org/pdf/2112.12870.pdf)]
      > This paper presents a new evaluation framework entitled Attributable to Identified Sources (AIS) for assessing the output of natural language generation models.
*   [2022/12] **Attributed Question Answering: Evaluation and Modeling for Attributed Large Language Models.** *B Bohnet et al. arXiv.* [[paper](https://arxiv.org/pdf/2212.08037.pdf)] [[code](https://github.com/google-research-datasets/Attributed-QA)]
*   [2023/04] **Evaluating Verifiability in Generative Search Engines** [[paper](http://arxiv.org/abs/2304.09848)]

*   [2023/05] **Evaluating and Modeling Attribution for Cross-Lingual Question Answering** *Benjamin Muller et al. arXiv.* [[paper](https://arxiv.org/abs/2305.14332)]
*   [2023/05] **FActScore: Fine-grained Atomic Evaluation of Factual Precision in Long Form Text Generation** [[paper](https://arxiv.org/abs/2305.14251)] [[code](https://github.com/shmsw25/FActScore)]
*   [2023/05] **"According to ..." Prompting Language Models Improves Quoting from Pre-Training Data** [[paper](https://arxiv.org/abs/2305.13252)]
      > This paper proposes according-to prompting to directing LLMs to ground responses against previously observed text, and propose QUIP-Score to measure the extent to which model-produced answers are directly found in underlying text corpora.
*   [2023/05] **Automatic Evaluation of Attribution by Large Language Models.** *X Yue et al. arXiv.* [[paper](https://arxiv.org/pdf/2305.06311.pdf)] [[code](https://github.com/OSU-NLP-Group/AttrScore)]
      > This paper investigate the automatic evaluation of attribution by LLMs - AttributionScore, by providing a definition of attribution and then explore two approaches for automatic evaluation. The results highlight both promising signals as well as remaining challenges for the automatic evaluation of attribution.
*   [2023/07] **FacTool: Factuality Detection in Generative AI -- A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios** [[paper](https://arxiv.org/abs/2307.13528v2)][[code](https://github.com/GAIR-NLP/factool)]


## 7. Limitations, Future Directions and Challenges in Fact Attribution.