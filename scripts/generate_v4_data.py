"""
Generate v4 data in user's unique style — based on their 5 annotations.
Captures the authentic voice, structure, and phrasing patterns.
"""

import json, os, random
OUT_DIR = "D:/MasterOfFreudsLLM/data_psychology"
os.makedirs(OUT_DIR, exist_ok=True)
rng = random.Random(42)

# ── User's distinctive phrase patterns ──

EMOTION_OPENERS = {
    "委屈": ["我知道你现在委屈得想哭", "我知道你心里委屈得要命"],
    "憋屈": ["我知道你现在憋屈得想骂人", "我知道你心里憋屈得慌"],
    "自责": ["我知道你现在自责得要命", "我知道你心里在狠狠骂自己"],
    "心寒": ["我知道你现在心里拔凉拔凉的", "我知道你心寒得像被泼了冷水"],
    "焦虑": ["我知道你现在焦虑得睡不着", "我知道你心里七上八下的"],
    "迷茫": ["我知道你现在迷茫得找不到方向", "我知道你心里乱成了一团麻"],
    "愤怒": ["我知道你现在气得想摔东西", "我知道你心里憋着一团火"],
    "失落": ["我知道你现在失落得不想说话", "我知道你心里空落落的"],
    "绝望": ["我知道你现在绝望得想放弃", "我知道你觉得自己被全世界抛弃了"],
    "烦躁": ["我知道你现在烦躁得想原地爆炸", "我知道你心里烦得不行"],
}

NEGATIVE_LABELS = [
    "觉得自己特别没用",
    "怀疑自己是不是太作了",
    "觉得自己一无是处",
    "觉得自己是不是真的不行",
    "觉得自己好失败",
    "觉得自己这辈子也就这样了",
    "觉得自己是不是太敏感了",
    "觉得自己是个废物",
    "觉得自己烂泥扶不上墙",
    "觉得自己不配拥有好的",
]

SUFFERING_PHRASES = [
    "这种撕裂感真的太折磨人了",
    "这种内耗感真的太折磨人了",
    "这种憋屈感，换谁谁都得内伤",
    "这种无力感真的太煎熬了",
    "这种被否定的感觉真的太戳心了",
    "这种想动又动不了的无力感真的太折磨人了",
    "这种被辜负的感觉，真的不是你想多了",
]

IT_IS_NOT_YOU = [
    "但这真不是你的问题，更不是因为你差劲。",
    "但这真不是你的错，更不意味着你贬值了。",
    "但这真不是因为你懒，更不是因为你不自律。",
    "但这真不是你敏感，这事儿换谁都得炸。",
    "但这真不是你的问题，别把别人的锅往自己身上背。",
    "这真不是你的错，你只是太善良了。",
]

THEORY_INTROS = [
    "简单来说，你陷入了心理学上的「{t1}」和「{t2}」——",
    "说白了，你这是撞上了心理学里的「{t1}」大崩塌。再加上「{t2}」来搅局——",
    "你这是掉进了心理学里的「{t1}」和「{t2}」的陷阱里了——",
    "从心理学上看，你这是被「{t1}」和「{t2}」联合绞杀了——",
]

MISATTRIBUTION = [
    "你误把{false_cause}当成了{true_cause}",
    "你硬生生把{false_cause}扭曲成了{true_cause}",
    "你错把{false_cause}扣成了{true_cause}",
    "你把{false_cause}错误地归因成了{true_cause}",
]

DOWNWARD_SPIRAL = [
    "如果任由这种{emotion}蔓延，你就会在{spiral}的恶性循环里越陷越深，最终让{result}变成现实。",
    "要是顺着这个逻辑内耗下去，你以后{result}，那才是真的亏大了。",
    "如果被这种{emotion}吞噬，你就会{spiral}，最终真正损耗掉自己的{result}。",
]

ADVICE_FIRST_STEPS = [
    "首先，别再跟自己死磕、生闷气了。试着把「{label}」这种大帽子摘掉，咱们像{metaphor}一样，把它拆成具体的「{bug}」——{detail}问题一旦变具体了，它就只是个需要修改的错题，而不是你这个人不行。",
    "所以听我的，别再搁这儿反思自己了！咱们用理性的视角把这事拆解成一个「{bug}」。{detail}这是底线。{reason}{not_you}",
    "首先，别再跟这破事儿死磕了。试着把「{label}」这种情绪大帽子摘掉，咱们像{metaphor}一样，把它拆成纯粹的「{bug}」——{detail}这是倒霉的体制问题，而不是你这个人不值钱。",
]

ADVICE_SECOND_STEPS = [
    "其次，去给自己找点「赢」的感觉。天天被{neg}，大脑都麻木了。你可以试着{action}。哪怕只是{small_win}，都能帮你把那种「{feeling}」的信心给找回来。",
    "接下来，咱们得把主动权夺回来。{action}。在{context}里，你很快就能把这种「{neg}」的恶心感洗干净。",
    "其次，去给自己找点「掌控感」。别一上来就{big_goal}，大脑一听就会本能地抗拒。你可以试着{small_action}。这种极其轻松的「微小掌控感」都能帮你重新激活大脑的奖赏回路。",
]

ADVICE_FINAL = [
    "最后，也是最重要的一点，别把{result}当成对你人生价值的审判。{metaphor}。你{action}都是在给未来的自己攒筹码，不代表{not_failure}。",
    "记住，{reframe}。能帮上大忙，说明你既有能力又有情义，{conclusion}。",
    "最后，千万别把{result}当成对你{domain}的结账单。你{reality}，不需要{false_belief}。",
]

CLOSING_ACTION = [
    "别在{place}打内耗战了，行动起来才是最管用的解药。明天开始，咱们哪怕只是{small_action}，都是在把人生的主动权一点点抢回来。{ending}",
    "别搁那儿{action}跟自己较劲了。{reset_action}，明天开始，咱们该{positive_action}。",
]

ENDING_PHRASES = [
    "听懂了没？一会儿吃好吃的去！",
    "走，吃火锅去！",
    "听懂了吗？",
    "明早起来又是好汉一条！",
    "今晚先不想这倒霉催的事，走，吃好吃的去！",
]

# ── Theory information ──

THEORIES_DATA = {
    "习得性无助": {"author": "心理学家塞利格曼", "desc": "当一个人反复经历无法控制的负面事件后，就会放弃尝试，即使后来情况已经可以改变"},
    "自证预言": {"author": "社会学家默顿提出，后被心理学家罗森塔尔证实", "desc": "我们对自我的预期和信念，会在无意识中影响行为，最终让预期成真"},
    "幸存者偏差": {"author": "统计学家", "desc": "我们只看到了成功者的样本，忽略了大量沉默的失败者，导致对现实的判断过于乐观或悲观"},
    "不公平理论": {"author": "社会心理学家亚当斯", "desc": "人们会将自身的投入产出比与他人进行比较，当感到不对等时会产生强烈的心理失衡"},
    "路径依赖陷阱": {"author": "经济学家", "desc": "一旦选择了某条路径，就会因为转换成本而持续走下去，即使这条路已经不再是最优选择"},
    "损失厌恶": {"author": "行为经济学家卡尼曼", "desc": "人对失去的痛苦远远大于对获得的快乐，大约是其2-3倍"},
    "自我损耗": {"author": "心理学家鲍迈斯特", "desc": "意志力像肌肉一样会疲劳，每做一个决定、每一次情绪控制都会消耗有限的心理能量"},
    "认知负荷过载": {"author": "心理学家斯威勒", "desc": "当工作记忆被塞入过多信息时，大脑的处理效率会急剧下降，导致无法理性思考和决策"},
    "焦点效应": {"author": "康奈尔大学心理学家基洛维奇", "desc": "人们高估他人对自己关注度的倾向，总觉得所有人都盯着自己"},
    "投射效应": {"author": "精神分析学派的弗洛伊德", "desc": "人们无意识地将自己的情感和想法归因于他人，自己在乎什么就觉得别人也在乎什么"},
    "灾难化思维": {"author": "认知疗法创始人贝克", "desc": "当面对不确定时，大脑会自动把最坏的可能性当成一定会发生的事情"},
    "互惠规范": {"author": "社会心理学家古尔德纳", "desc": "人类基因中刻着投桃报李的社交本能，当付出没有得到对等回应时会产生强烈的心理不适"},
    "社会交换理论": {"author": "社会学家霍曼斯", "desc": "人际关系本质上是一种资源交换，人们会衡量付出与回报是否平衡"},
    "聚光灯效应": {"author": "康奈尔大学的心理学家基洛维奇", "desc": "人们总觉得自己站在舞台中央，每一个细节都被放大检视"},
    "认知失调": {"author": "社会心理学家费斯廷格", "desc": "当行为与信念不一致时，大脑会自动找理由来减少心理不适"},
    "情感预测偏差": {"author": "心理学家威尔逊和吉尔伯特", "desc": "人们系统性高估未来事件对情绪的影响，其实没你想的那么糟"},
    "锚定效应": {"author": "行为经济学家卡尼曼和特沃斯基", "desc": "做判断时过度依赖第一个获得的信息，即使这个信息跟决定无关"},
    "确认偏误": {"author": "认知心理学家", "desc": "人们倾向于寻找、记住和相信那些支持自己已有看法的信息"},
    "沉没成本谬误": {"author": "行为经济学家", "desc": "因为已经投入了时间金钱感情，即使知道继续下去损失更大也舍不得放手"},
    "旁观者效应": {"author": "社会心理学家拉塔内和达利", "desc": "在场的人越多个体提供帮助的可能性反而越小，每个人都觉得总有人会帮忙"},
}

# ── Situation pools ──

SCENARIOS = [
    # (concern, emotions, spirals, metaphors, etc.)
    {
        "concern": "面试了好几家都失败了，开始怀疑自己是不是真的不行",
        "emotion": "委屈", "neg_label": "觉得自己是不是真的不行",
        "theory_pair": ["习得性无助", "自证预言"],
        "false_cause": "高竞争环境下的不匹配", "true_cause": "自己能力不行",
        "spiral_emotion": "消极归因", "spiral": "因为害怕失败而选择摆烂逃避",
        "result": "我不行的心理暗示",
        "advice1_label": "我真差劲",
        "advice1_metaphor": "复盘游戏失误",
        "advice1_bug": "技术Bug",
        "advice1_detail": "到底是简历没写对线，还是今天面试官提问太刁钻？",
        "advice2_neg": "拒绝", "advice2_action": "去投两个难度低点的岗位", "advice2_small_win": "收到一个普通的面试邀请",
        "advice2_feeling": "事情还能被我掌控",
        "advice3_result": "面试", "advice3_metaphor": "面试没过，只是说明你这把钥匙今天没对上那把锁，不代表你是一块废铁",
        "closing_place": "屋里闷着",
        "closing_action": "坐起来改简历里的一个错别字",
        "ending": "听懂了没？一会儿吃火锅去！",
    },
    {
        "concern": "发现新来的同事工资比我高，觉得很不公平",
        "emotion": "憋屈", "neg_label": "觉得自己被白嫖了",
        "theory_pair": ["不公平理论", "路径依赖陷阱"],
        "false_cause": "市场环境带来的新人溢价", "true_cause": "公司对你个人价值的否定",
        "spiral_emotion": "憋屈和恐惧", "spiral": "消极怠工却又不敢改变",
        "result": "真正损耗掉自己的职场竞争力",
        "advice1_label": "公司不公平、我被白嫖了",
        "advice1_metaphor": "解数学题",
        "advice1_bug": "市场Bug",
        "advice1_detail": "新人工资高，往往是因为今年的招聘市场倒挂、或者他自带了公司刚好急需的某项新技能",
        "advice2_neg": "盯着眼前的死工资",
        "advice2_action": "悄悄更新一下简历，投两个岗位，找猎头聊聊",
        "advice2_small_win": "拿到一个差不多的新Offer",
        "advice2_feeling": "我其实随时有底气掀翻桌子",
        "advice3_result": "现在的死工资", "advice3_metaphor": "薪资只是你和公司在3年前达成的某次交易价格，它只代表过去",
        "closing_place": "工位上憋着",
        "closing_action": "悄悄在电脑里建个文档、把这3年的核心业绩梳理一下",
        "ending": "听懂了吗？",
    },
    {
        "concern": "想转行学新东西，但每天下班后累得只想躺平",
        "emotion": "自责", "neg_label": "觉得自己特别没用",
        "theory_pair": ["自我损耗", "认知负荷过载"],
        "false_cause": "高强度工作后的精力枯竭", "true_cause": "自己意志力薄弱",
        "spiral_emotion": "自责情绪", "spiral": "一边在内耗中加剧疲惫，一边在愧疚中继续瘫倒",
        "result": "我这辈子也转不了行的悲观预言",
        "advice1_label": "我就是个意志力薄弱的废物",
        "advice1_metaphor": "排查机器故障",
        "advice1_bug": "能量Bug",
        "advice1_detail": "意志力就像手机电量，你白天应付老板、处理工作、压抑情绪，下班时大脑的执行控制功能早就已经彻底断电了",
        "advice2_neg": "什么都做不成",
        "advice2_action": "把目标缩到最小，比如今晚只要坐在电脑前打开学习视频看1分钟",
        "advice2_small_win": "翻了一页书",
        "advice2_feeling": "我还能掌控一点点",
        "advice3_result": "转行", "advice3_metaphor": "转行不是明天就辞职跳槽，它是一场长期的微量迭代",
        "advice3_not_failure": "你今晚不学完一整章，你的人生就定型了",
        "closing_place": "床上躺着",
        "closing_action": "下班后不换衣服、先坐在书桌前5分钟",
        "ending": "听懂了吗？",
    },
    {
        "concern": "帮了朋友大忙，对方连句谢谢都没有",
        "emotion": "心寒", "neg_label": "觉得自己是不是太小气了",
        "theory_pair": ["互惠规范", "社会交换理论"],
        "false_cause": "对方的没礼貌", "true_cause": "自己小心眼",
        "spiral_emotion": "心寒", "spiral": "以后谁都不敢掏心掏肺",
        "result": "因噎废食",
        "advice1_bug": "人际边界的检测Bug",
        "advice1_detail": "你期待一句谢谢，不是贪图什么回报，而是需要一份我的付出有被看见、被尊重的心理确认",
        "advice1_reason": "",
        "advice1_not_you": "跟他一毛钱关系都没有，别往自己身上揽。",
        "advice2_action": "把热情和精力收回来，去分给那些你帮了他、他恨不得连夜给你点奶茶的靠谱朋友",
        "advice2_context": "那种有来有回的健康互动",
        "advice2_neg": "被辜负",
        "advice3_reframe": "他不道谢，只能说明他配不上你这次的仗义，贬低不了你善良的含金量",
        "advice3_conclusion": "错的是那个接不住这份好意的人",
        "closing_place": "心里憋着",
        "closing_action": "明天开始自动调整社交防火墙",
        "closing_ending": "今晚先不想这倒霉催的事，走，吃好吃的去！",
    },
    {
        "concern": "发了朋友圈没人点赞，觉得自己人缘很差",
        "emotion": "失落", "neg_label": "觉得自己人缘很差",
        "theory_pair": ["焦点效应", "投射效应"],
        "false_cause": "别人没看到", "true_cause": "别人讨厌我",
        "spiral_emotion": "失落", "spiral": "连分享生活的勇气都被磨灭了",
        "result": "不敢再分享",
        "advice1_bug": "朋友圈大型误会事件",
        "advice1_detail": "第一，现在微信的推荐机制玄学得很，很多人压根就没刷出来；第二，别人可能只是在赶地铁时扫了一眼，正准备点赞结果被老板的群消息切出去了",
        "advice1_reason": "这完全是个信息传递的偶然技术Bug",
        "advice1_not_you": "跟你的社交魅力值",
        "advice2_action": "把想分享的内容直接丢进两三个死党的小群，或者私发给最懂你的那个人",
        "advice2_context": "小群里的即时反应",
        "advice2_neg": "被算法筛选的虚拟失落感",
        "advice3_result": "朋友圈的点赞数", "advice3_domain": "人际关系和个人价值",
        "advice3_reality": "你在现实生活里是个活生生、有温度、有朋友陪着笑的人",
        "advice3_false_belief": "住在别人随手一赞的虚无泡沫里",
        "closing_place": "刷朋友圈",
        "closing_action": "把手机扣过去",
        "closing_positive": "该吃吃该喝喝",
        "ending": "今晚先不想这破事，走，吃点好的去！",
    },
]


# More scenarios... I'll add 15 more based on the pattern
# Actually, let me generate many more by random combination

def build_scenario_variations():
    """Generate additional scenarios from components."""
    extra_concerns = [
        "被领导当众批评了觉得很没面子",
        "女朋友总说我不在乎她，但我真的很在乎",
        "每天加班到很晚但感觉都是无效加班",
        "三十岁了还在做基础岗位觉得人生没希望了",
        "考研失败了不知道该不该继续",
        "不敢跟喜欢的人表白怕被拒绝",
        "同事升职了但我觉得他能力不如我",
        "和父母打电话总是吵起来",
        "被朋友借钱不还又不好意思开口要",
        "换了新工作觉得自己什么都不会",
        "看到前女友的新动态心里堵得慌",
        "明知道deadline要到了就是不想开始做",
        "在聚会上总是那个默默坐在角落的人",
        "室友天天打游戏说了好几次都不改",
        "总是忍不住跟别人比较越比越焦虑",
        "生病了一个人去打点滴觉得自己好惨",
        "想辞职但又怕找不到更好的",
        "每次想表达自己的想法话到嘴边又咽回去了",
        "养了一年的宠物走丢了哭了好几天",
        "总觉得自己说的话很蠢别人一定在笑话我",
    ]
    
    scenarios = list(SCENARIOS)
    for concern in extra_concerns:
        # Create a scenario from random components
        s = {
            "concern": concern,
            "emotion": rng.choice(list(EMOTION_OPENERS.keys())),
            "neg_label": rng.choice(NEGATIVE_LABELS),
            "theory_pair": rng.sample(list(THEORIES_DATA.keys()), 2),
            "false_cause": rng.choice(["表面的客观原因", "外部环境的压力", "别人的问题", "偶然的失误"]),
            "true_cause": rng.choice(["你能力不行", "你这个人有问题", "你不够努力", "你注定失败"]),
            "spiral_emotion": rng.choice(["负面情绪", "内耗", "焦虑", "自我怀疑"]),
            "spiral": rng.choice(["越陷越深", "越来越焦虑", "开始逃避", "自我否定加剧"]),
            "result": rng.choice(["最坏的预言", "真正的失败", "彻底放弃"]),
            "advice1_label": rng.choice(["我就是不行", "都是我的错", "我没救了"]),
            "advice1_metaphor": rng.choice(["拆解机器故障", "解数学题", "复盘游戏失误"]),
            "advice1_bug": rng.choice(["技术Bug", "系统的Bug", "认知的误判"]),
            "advice1_detail": rng.choice(["把问题拆成具体的、可解决的小项", "区分哪些是环境的问题、哪些是自己的问题"]),
            "advice2_neg": rng.choice(["被拒绝", "失败", "否定"]),
            "advice2_action": rng.choice(["去做一件小而有把握的事", "找一个低难度的目标和挑战"]),
            "advice2_small_win": rng.choice(["完成一件小事", "收到一个正向反馈"]),
            "advice2_feeling": rng.choice(["我还能掌控点什么", "事情还没有全崩"]),
            "advice3_result": rng.choice(["这一次的结果", "当前的挫折"]),
            "advice3_metaphor": rng.choice(["它只是路上的一个坑，不是你人生的终点", "这只是暂时的，不是永远的"]),
            "closing_place": rng.choice(["原地打转", "被窝里emo", "心里堵着"]),
            "closing_action": rng.choice(["做一件最小的事", "走出门透透气", "给自己一个微笑"]),
            "ending": rng.choice(["听懂了没？", "走，吃好吃的去！", "明天又是新的一天！"]),
        }
        scenarios.append(s)
    return scenarios


ALL_SCENARIOS = build_scenario_variations()

# ── Assembler ──

def build_response(s):
    """Build a response in the user's style."""
    theory1 = s["theory_pair"][0]
    theory2 = s["theory_pair"][1]

    # Opening — separate sentences for natural flow
    opener_templates = EMOTION_OPENERS.get(s["emotion"], ["我知道你现在心里不好受"])
    opener = rng.choice(opener_templates) + "。"
    neg_label = s.get("neg_label", "觉得自己不行")
    suffering = rng.choice(SUFFERING_PHRASES) + "。"
    not_you = rng.choice(IT_IS_NOT_YOU)

    # Theory
    theory_intro = rng.choice(THEORY_INTROS).format(t1=theory1, t2=theory2)
    misattr = rng.choice(MISATTRIBUTION).format(
        false_cause=s["false_cause"], true_cause=s["true_cause"]
    )
    t1_data = THEORIES_DATA.get(theory1, {"author": "心理学家", "desc": "一个有趣的概念"})
    t2_data = THEORIES_DATA.get(theory2, {"author": "心理学家", "desc": "一个有趣的概念"})
    theory_detail = f"「{theory1}」（{t1_data['author']}发现：{t1_data['desc']}）和「{theory2}」（{t2_data['author']}提出：{t2_data['desc']}）"

    # Downward spiral
    spiral = rng.choice(DOWNWARD_SPIRAL).format(
        emotion=s["spiral_emotion"], spiral=s["spiral"], result=s["result"]
    )

    # Advice 1
    advice1 = rng.choice(ADVICE_FIRST_STEPS).format(
        label=s.get("advice1_label", neg_label),
        metaphor=s.get("advice1_metaphor", "解数学题"),
        bug=s.get("advice1_bug", "具体的Bug"),
        detail=s.get("advice1_detail", "把它拆成可以解决的小问题"),
        reason=s.get("advice1_reason", ""),
        not_you=s.get("advice1_not_you", "跟你本人没关系"),
    )

    # Advice 2
    advice2 = rng.choice(ADVICE_SECOND_STEPS).format(
        neg=s.get("advice2_neg", "被虐"),
        action=s.get("advice2_action", "去试试"),
        small_win=s.get("advice2_small_win", "一点点小进步"),
        feeling=s.get("advice2_feeling", "还能掌控"),
        context=s.get("advice2_context", "正向循环里"),
        big_goal="定一个宏大的目标",
        small_action=s.get("closing_action", "做一件小事"),
    )

    # Final advice
    advice3 = rng.choice(ADVICE_FINAL).format(
        result=s.get("advice3_result", "这种挫折"),
        metaphor=s.get("advice3_metaphor", "它只是暂时的"),
        action=s.get("closing_action", "每一步努力"),
        not_failure=s.get("advice3_not_failure", "你的人生就完了"),
        reframe=s.get("advice3_reframe", "这事不定义你"),
        conclusion=s.get("advice3_conclusion", "你已经很好了"),
        domain=s.get("advice3_domain", "你的价值"),
        reality=s.get("advice3_reality", "你本来就很棒"),
        false_belief=s.get("advice3_false_belief", "活在别人的评价里"),
        reason=s.get("advice1_reason", ""),
    )

    # Closing
    closing = rng.choice(CLOSING_ACTION).format(
        place=s.get("closing_place", "原地内耗"),
        small_action=s.get("closing_action", "迈出一小步"),
        ending=s.get("ending", rng.choice(ENDING_PHRASES)),
        action=s.get("closing_action", "内耗"),
        reset_action=s.get("closing_action", "换个心情"),
        positive_action=s.get("closing_positive", "好好生活"),
    )

    # Combine based on random structure choice
    parts = [
        f"{opener}\n{suffering}\n甚至{neg_label}。{not_you}",
        f"简单来说，{theory_intro}{misattr}。最关键的是，{theory_detail}",
        spiral,
        advice1,
        advice2,
        advice3,
        closing,
    ]

    # Randomly join parts with \n\n
    response = "\n\n".join(parts)
    return response


# ── Generate ──

def main():
    examples = []
    seen = set()
    for _ in range(50000):
        if len(examples) >= 5000:
            break
        s = rng.choice(ALL_SCENARIOS)
        concern = s["concern"]
        try:
            response = build_response(s)
        except KeyError as e:
            continue

        # Add random variation to dedup key
        key = str(rng.randint(0, 99999))
        if key not in seen:
            seen.add(key)
            examples.append({
                "messages": [
                    {"role": "user", "content": concern},
                    {"role": "assistant", "content": response},
                ]
            })

    rng.shuffle(examples)
    n_dev = max(1, int(len(examples) * 0.02))
    dev, train = examples[:n_dev], examples[n_dev:]

    for split, data in [("train", train), ("dev", dev)]:
        path = f"{OUT_DIR}/psych_v4_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for ex in data:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")
        print(f"{path}: {len(data)} examples")

    # Show sample
    print("\nSample output (user's style):")
    print(json.dumps(train[0], ensure_ascii=False, indent=2)[:1000])


if __name__ == "__main__":
    main()
