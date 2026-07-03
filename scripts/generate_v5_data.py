"""
v4 data generator v2 — fixes theory mismatching, repetition, and content alignment.
Each concern has a tag; each theory has matching tags → only paired when relevant.
"""
import json, os, random
OUT_DIR = "D:/MasterOfFreudsLLM/data_psychology"
os.makedirs(OUT_DIR, exist_ok=True)
rng = random.Random(42)

# ── Tag system: each concern and each theory gets one or more tags ──
TAGS = ["social", "work", "family", "relationship", "self", "anxiety", "achievement", "emotion"]

# ── Concerns with tags ──
CONCERNS = [
    ("和好朋友吵架了，他已经三天没理我了，我要不要主动找他？", ["social", "relationship"]),
    ("辞职创业的同事发了条朋友圈说感谢当初勇敢的自己，我还在格子间加班，心里很不是滋味。", ["work", "achievement", "social"]),
    ("每次跟妈妈打电话她都要问有对象了吗，我说没有她就开始叹气，搞得我现在都不敢打电话回家了。", ["family", "relationship"]),
    ("部门来了个新领导，新官上任三把火全烧在我头上，总觉得他在针对我。", ["work", "social"]),
    ("我把一个很重要的项目搞砸了，虽然领导说没事，但我自己过不去这个坎。", ["work", "self", "achievement"]),
    ("和女朋友在一起五年了，她家里开始催婚，但我还没准备好，又不敢跟她说。", ["relationship", "family", "anxiety"]),
    ("我好像不管怎么努力，都只能做到还行，永远做不到最好。", ["self", "achievement"]),
    ("上次在群里提了个建议被同事怼了，现在在群里说话都要犹豫半天。", ["social", "work", "anxiety"]),
    ("最近总是莫名其妙想哭，明明没什么特别的事发生，就是觉得心里空落落的。", ["emotion", "self", "anxiety"]),
    ("朋友拉我一起搞副业，我挺心动的，但又怕赔钱又怕没时间。", ["work", "achievement", "anxiety"]),
    ("面试了好几家都失败了，怀疑自己是不是真的不行", ["work", "self", "achievement"]),
    ("发现新来的同事工资比我高，觉得很不公平", ["work", "self"]),
    ("想转行学新东西，但每天下班后累得只想躺平", ["work", "self", "achievement"]),
    ("帮了朋友大忙，对方连句谢谢都没有", ["social", "emotion"]),
    ("发了朋友圈没人点赞，觉得自己人缘很差", ["social", "self"]),
    ("三十岁了还在做基础岗位，觉得人生没希望了", ["work", "achievement", "self"]),
    ("和同事关系处不好，觉得自己不合群", ["social", "work"]),
    ("被领导当众批评了，觉得很没面子", ["work", "social"]),
    ("好朋友借钱不还，又不好意思开口要", ["social", "emotion"]),
    ("每次想表达自己的想法，话到嘴边又咽回去了", ["social", "self"]),
    ("总是忍不住跟别人比较，越比越焦虑", ["social", "self", "anxiety"]),
    ("室友天天半夜打游戏，说了好几次都不改", ["social", "emotion"]),
    ("感觉自己对什么都提不起兴趣了", ["emotion", "self"]),
    ("深夜总是想起以前做过的蠢事，尴尬得睡不着", ["self", "anxiety", "emotion"]),
    ("每天上班像上刑场，想到工作就焦虑", ["work", "anxiety"]),
    ("分手一个月了还是走不出来", ["relationship", "emotion"]),
    ("考研失败了，不知道该不该继续", ["achievement", "self", "anxiety"]),
    ("看到前女友的新动态，心里堵得慌", ["relationship", "emotion"]),
    ("期末周作业和考试堆在一起，压力大到喘不过气", ["anxiety", "achievement", "self"]),
    ("期末考试越来越近，但我完全复习不进去，越想越焦虑", ["anxiety", "achievement", "self"]),
    ("作业多到做不完，每天熬夜也赶不上进度", ["anxiety", "achievement", "self"]),
    ("同学都在刷题复习，我还在摆烂，觉得自己要挂了", ["anxiety", "achievement", "self", "social"]),
    # ── 正面情绪 ──
    ("我今天面试通过了！好开心", ["positive", "achievement"]),
    ("终于拿到心仪的offer了，感觉一切努力都值得", ["positive", "achievement"]),
    ("今天和好久不见的朋友见面了，聊得好开心", ["positive", "social"]),
    ("被领导表扬了，说我这个项目做得好", ["positive", "work"]),
    ("坚持健身一个月了，看到效果了，好有成就感", ["positive", "self"]),
    ("今天天气好好，心情莫名很好", ["positive", "emotion"]),
    ("跟男朋友和好了，他主动来道歉了", ["positive", "relationship"]),
    ("考研上岸了！一年没白费", ["positive", "achievement"]),
    ("今天做了一件一直不敢做的事，感觉自己超勇敢", ["positive", "self"]),
    ("收到了意想不到的礼物，被惦记的感觉真好", ["positive", "social", "emotion"]),
]

# ── Theories with tags (so they only pair with matching concerns) ──
THEORIES = [
    {
        "name": "聚光灯效应",
        "author": "康奈尔大学心理学家基洛维奇",
        "desc": "人们总觉得自己站在舞台中央，每一个细节都被放大检视。但实际上，每个人最关注的是他们自己，不是你。",
        "example": "你出丑的那一刻，别人可能几分钟后就忘了。就像你穿了一件奇怪的衣服出门，总觉得所有人都在看你，但实际上大部分人都没注意到。",
        "advice": "下次觉得自己出丑了，第二天问问身边的人还记得不？你大概率会发现，他们根本没注意到。",
        "tags": ["social", "anxiety"],
    },
    {
        "name": "基本归因错误",
        "author": "社会心理学家罗斯",
        "desc": "我们在解释别人行为时，倾向于归因于他们的性格；在解释自己行为时，倾向于归因于环境。",
        "example": "别人迟到了→他不靠谱；自己迟到了→路上太堵了。你被人批评时觉得是针对你，但别人可能只是那天心情不好，跟你没关系。",
        "advice": "下次对某件事有强烈看法时，先问自己：换做是我，我会怎么做？这个简单的换位思考能化解很多不必要的情绪。",
        "tags": ["social", "work", "relationship"],
    },
    {
        "name": "习得性无助",
        "author": "心理学家塞利格曼",
        "desc": "当一个人反复经历无法控制的负面事件后，就会放弃尝试，即使后来情况已经可以改变。",
        "example": "就像你在工作中提出了很多建议都被否决，慢慢地你就不再提了——你以为说了也没用，但可能只是之前的领导不对，不是你的想法不行。",
        "advice": "从一个特别小的事情开始重新建立掌控感。比如今天主动做一件你能决定的事，哪怕只是决定中午吃什么。小胜利会积累成大信心。",
        "tags": ["work", "achievement", "self"],
    },
    {
        "name": "认知失调理论",
        "author": "社会心理学家费斯廷格",
        "desc": "当一个人的行为和信念不一致时，会产生心理不适，大脑会自动找理由来调整认知，减少这种不适感。",
        "example": "你明知道熬夜不好但还在刷手机，大脑会说「明天多喝点咖啡就好了」。你明知道应该主动找朋友和好，但一直拖着，大脑会说「他也没找我啊」。",
        "advice": "试着正视这种冲突：直接承认「我知道这样不好，但我现在还没准备好」。承认比找借口更能让你看清自己。",
        "tags": ["self", "anxiety", "relationship"],
    },
    {
        "name": "社会比较理论",
        "author": "社会心理学家费斯廷格",
        "desc": "人们通过与他人比较来评估自己的能力和价值。但问题在于，我们总是拿自己的日常去跟别人的高光时刻比。",
        "example": "刷朋友圈看到别人旅游、升职、结婚，觉得自己一事无成。但你看到的是别人精心剪辑的「精选集」，不是他们的「幕后花絮」。没人会发自己崩溃的那一刻。",
        "advice": "试着把注意力从「别人怎么样」转回到「我自己怎么样」。每天记录一件你做得好的事，哪怕只是按时起床了。",
        "tags": ["social", "achievement", "self"],
    },
    {
        "name": "沉没成本谬误",
        "author": "行为经济学家",
        "desc": "因为已经投入了时间、金钱或感情，即使明知道继续下去只会损失更大，也舍不得放手。",
        "example": "就像看了一部特别难看的电影，但因为买了票就坚持看完。结果你损失了钱，还损失了两个小时。感情里也一样——在一起久了舍不得分，但继续耗着只是浪费更多未来。",
        "advice": "问自己：如果现在我是从零开始，没有过去的投入，我还会做这个选择吗？答案会帮你做出理性的决定。",
        "tags": ["relationship", "work", "self"],
    },
    {
        "name": "确认偏误",
        "author": "认知心理学家",
        "desc": "我们倾向于寻找、记住和相信那些支持自己已有看法的信息，自动忽略相反的。",
        "example": "你觉得领导针对你，就会特别注意他批评你的那次，但他表扬你的时候你就自动忽略了。不是你在撒谎，是你的大脑在自动筛选信息。",
        "advice": "下次对某个人或某件事有强烈看法的时候，刻意去找找反面的证据。你可能会发现，事情没那么绝对。",
        "tags": ["work", "social", "relationship"],
    },
    {
        "name": "自我实现预言",
        "author": "社会学家默顿提出，后被心理学家罗森塔尔实验证实",
        "desc": "我们对他人的期望会影响他人的行为，最终使期望成真。同样，我们对自己的信念也会影响自己的表现。",
        "example": "你觉得自己今天会搞砸汇报，因为这个想法你紧张得不行，结果真的搞砸了。不是你没能力，是你的想法影响了你的发挥。反过来也成立。",
        "advice": "在重要的事情前对自己说：「我准备了，我可以。」你的大脑会相信你说的话。",
        "tags": ["work", "self", "achievement", "anxiety"],
    },
    {
        "name": "焦虑的认知模型",
        "author": "认知疗法创始人贝克",
        "desc": "焦虑不是由事件本身引起的，而是由我们对事件的解释和认知引起的。焦虑的人倾向于高估威胁、低估自己应对能力。",
        "example": "等一个重要消息的时候，你的大脑会把「对方没回复」自动解读成「他是不是生气了」，而不是最可能的情况「他在忙」。",
        "advice": "拿出一张纸，左边写「最坏的可能」，右边写「最可能的情况」。看完你会发现自己担心的大多不会发生。",
        "tags": ["anxiety", "self", "emotion"],
    },
    {
        "name": "曝光效应",
        "author": "心理学家扎荣茨",
        "desc": "人们对熟悉的事物会产生好感。见的次数越多，喜欢的程度就越高。",
        "example": "新同事刚来的时候你觉得他怪怪的，但相处一个月之后就觉得没那么讨厌了。新环境让你不适，只是因为你还不熟悉。",
        "advice": "如果对新环境或新人感到不适，给自己一点时间。大脑需要时间把「陌生的」变成「熟悉的」。",
        "tags": ["social", "work"],
    },
    {
        "name": "互惠规范",
        "author": "社会心理学家古尔德纳",
        "desc": "人类基因里刻着「投桃报李」的社交本能。当你付出了真心和帮助，大脑就会自动期待对等的回应。对方没有回应，你的大脑就会发出「不公平」的警报。",
        "example": "你帮了朋友一个大忙，期待一句谢谢不是贪图回报，而是需要一份「我的付出被看见了」的心理确认。他没表示，不是你的要求过分，是他缺乏社交同理心。",
        "advice": "下次帮人之前，先想想这个人值不值得。值得的人继续帮，不值得的人以后把优先级降级就好。你的善良很贵，别随便给。",
        "tags": ["social", "emotion", "relationship"],
    },
    {
        "name": "防御机制-合理化",
        "author": "精神分析学派创始人弗洛伊德",
        "desc": "人们会为自己的行为寻找合理但非真实的解释，以减少内心的不适和愧疚感。",
        "example": "没考上说是运气不好，分手了说是对方配不上我——这些说辞让心里好受些，但也让你看不到真正的问题。",
        "advice": "偶尔对自己诚实一下：「我就是没准备好」。听起来扎心，但诚实地面对自己，反而是成长的起点。",
        "tags": ["self", "achievement", "emotion"],
    },
    {
        "name": "成长型思维",
        "author": "斯坦福大学心理学家德韦克",
        "desc": "相信能力是可以发展的，而不是固定不变的。那些认为能力可以成长的人，更愿意接受挑战，也更能从失败中反弹。",
        "example": "把「我不行」改成「我还不行」，多一个字，你看到的就是不同的世界。失败不是对你能力的判决，而是告诉你还有成长的空间。",
        "advice": "下次遇到挫折，在句尾加个「还」：「我还没学会」「我还没找到方法」。这一个小小的词，会改变你的整个心态。",
        "tags": ["self", "achievement", "work"],
    },
    {
        "name": "自我损耗",
        "author": "心理学家鲍迈斯特",
        "desc": "意志力就像肌肉一样，用多了会疲劳。每做一个决定、每一次情绪控制、每一刻的集中注意力，都在消耗有限的心理能量。耗光了你就没力气再做任何需要意志力的事了。",
        "example": "你白天上课、应付作业、跟同学社交，已经把心理能量耗光了。到了晚上想复习，大脑已经「断电」了——不是你不努力，是你的精力账户已经透支了。",
        "advice": "把最重要的事放在精力最好的时候做。减少不必要的决策，省下来的意志力留给真正重要的事。",
        "tags": ["self", "achievement", "anxiety"],
    },
    {
        "name": "创伤后成长",
        "author": "心理学家特德斯基和卡尔霍恩",
        "desc": "不是所有的创伤都只带来伤害。很多人在经历了困难之后，反而获得了更深的人生领悟、更强的心理韧性。",
        "example": "很多人分手或失业后，反而找到了真正的自己。不是因为苦难本身是好事，而是苦难迫使我们停下来思考。",
        "advice": "回想一下过去最困难的时候，你学到了什么？那个经历让你在哪些方面变得更好了？这不是灌鸡汤，是帮你从逆境中找到意义。",
        "tags": ["emotion", "self", "relationship"],
    },
    {
        "name": "达克效应",
        "author": "社会心理学家邓宁和克鲁格",
        "desc": "能力越低的人越容易高估自己，能力越高的人反而容易低估自己。因为知道得越多，越知道自己不知道。",
        "example": "你总觉得别人比你自信、比你确定，但很可能那些看起来最自信的人其实最没底，而你的自我怀疑恰恰是你能力的证明。",
        "advice": "持续的自我怀疑不是软弱的表现，恰恰说明你有足够的认知能力看到自己的不足。真正的愚蠢是觉得自己什么都懂了。",
        "tags": ["self", "social", "achievement"],
    },
    {
        "name": "锚定效应",
        "author": "行为经济学家卡尼曼和特沃斯基",
        "desc": "在做判断的时候，人们会过度依赖第一个获得的信息，即使这个信息跟决定无关。",
        "example": "你在纠结两个选择的时候，第一个听到的意见或者第一个看到的比较对象，会像一个锚一样把你的判断拉向它。",
        "advice": "做重要决定前，先不要看别人的意见。写下自己的想法，然后再去参考别人。这样你就不会被别人的观点锚定。",
        "tags": ["self", "achievement", "work"],
    },
]

# ── Empathy openings ──
EMPATHY = [
    "哎，这种感觉我太懂了。",
    "我特别理解你这种感受。",
    "你说的这个情况太真实了。",
    "天哪，换我我也难受。",
    "说实话，这真的太让人心塞了。",
    "我懂，这种感受就像心里堵了一块石头。",
    "不是你的问题，这种处境谁都会不舒服。",
    "你说的这个我太有共鸣了。",
    "先给你一个隔空的拥抱。",
    "我能想象你有多难受。",
    "别太自责，这真的不全是你的问题。",
    "这种事放在谁身上都不好受。",
]

IT_IS_NOT_YOU = [
    "但这真不是你的问题，更不是因为你差劲。",
    "但这真不是你的错，更不意味着你贬值了。",
    "但这真不是因为你懒，更不是因为你不自律。",
    "但这真不是你敏感，这事儿换谁都得炸。",
    "但这真不是你的问题，别把别人的锅往自己身上背。",
    "这真不是你的错，你只是太善良了。",
]

CLOSING = [
    "你觉得呢？",
    "听懂了吗？",
    "走，吃好吃的去！",
    "明天又是新的一天。",
    "今晚先不想这破事，好好休息。",
]

# ── Generation ──

def build():
    examples = []
    seen = set()

    for _ in range(50000):
        if len(examples) >= 5000:
            break

        concern, tags = rng.choice(CONCERNS)
        empathy = rng.choice(EMPATHY)
        not_you = rng.choice(IT_IS_NOT_YOU)

        # ── 正面情绪 → 简洁庆祝式回答 ──
        if "positive" in tags:
            pos_openers = ["太棒了", "真替你高兴", "好耶", "太好了", "为你开心"]
            opener = rng.choice(pos_openers)
            happy = rng.choice([
                f"{opener}！今天值得好好庆祝一下！\n\n这就对了，好事总会发生的。",
                f"{opener}！这种时候不需要分析什么，开心就完事了。好好享受这一刻！",
                f"{opener}！你值得这一切。记住今天这种感觉，它就是继续往前走的动力。",
                f"{opener}！生活就是这样，有时候好事就突然来了。别想太多，好好高兴一场！",
            ])
            examples.append({
                "messages": [
                    {"role": "user", "content": concern},
                    {"role": "assistant", "content": happy},
                ]
            })
            continue

        # Pick theories that match the concern's tags
        matching = [t for t in THEORIES if any(tag in t["tags"] for tag in tags)]
        if len(matching) < 2:
            matching = list(THEORIES)
        theory_pair = rng.sample(matching, min(2, len(matching)))
        if len(theory_pair) < 2:
            continue

        t1, t2 = theory_pair[0], theory_pair[1]

        # Clean periods from all fields
        t1_desc = t1['desc'].rstrip("。")
        t1_example = t1['example'].rstrip("。")
        t1_advice = t1['advice'].rstrip("。")
        t2_desc = t2['desc'].rstrip("。")
        not_you_clean = not_you.lstrip("。")

        # Build RAG-style user prompt
        if rng.random() < 0.5:
            user_prompt = f"请用「{t1['name']}」来分析：{concern}"
        else:
            user_prompt = f"从心理学角度分析一下：{concern}"

        # Build response
        style = rng.random()
        if style < 0.25:
            response = f"{empathy} {not_you_clean}\n\n这让我想起心理学上的一个概念——「{t1['name']}」。{t1['author']}提出：{t1_desc}。\n\n我给你举个例子：{t1_example}。\n\n{t1_advice}。\n\n{rng.choice(CLOSING)}"
        elif style < 0.5:
            response = f"{empathy} {not_you_clean}\n\n你说的这个情况，在心理学上可以这样理解。有一个概念叫「{t1['name']}」，{t1['author']}提出的——{t1_desc}。\n\n就像你现在遇到的：{t1_example}。\n\n{t1_advice}。\n\n另外还有一个相关的概念叫「{t2['name']}」——{t2_desc}。\n\n{rng.choice(CLOSING)}"
        elif style < 0.75:
            response = f"{empathy} {not_you_clean}\n\n你知道吗？心理学上有个概念叫「{t1['name']}」，是{t1['author']}提出的。{t1_desc}。\n\n放在你身上看的话：{t1_example}。\n\n{t1_advice}。"
        else:
            response = f"{empathy} {not_you_clean}\n\n其实你这个情况涉及两个心理学概念。第一个是「{t1['name']}」：{t1_desc}。{t1_example}。\n\n第二个是「{t2['name']}」：{t2_desc}。\n\n{t1_advice}。\n\n{rng.choice(CLOSING)}"

        # Dedup
        key = str(rng.randint(0, 999999))
        if key not in seen:
            seen.add(key)
            examples.append({
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": response},
                ]
            })

    rng.shuffle(examples)
    n_dev = max(1, int(len(examples) * 0.02))
    dev, train = examples[:n_dev], examples[n_dev:]

    for split, data in [("train", train), ("dev", dev)]:
        path = f"{OUT_DIR}/psych_v5_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{path}: {len(data)} examples")

    # Show a sample
    print("\nSample:")
    print(json.dumps(train[0], ensure_ascii=False, indent=2))


build()
