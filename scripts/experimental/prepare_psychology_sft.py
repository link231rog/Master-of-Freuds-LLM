"""
Prepare psychology-focused SFT data for Master of Freud's LLM.
Friend-style: sounds like a wise friend, not a psychologist giving a lecture.

Generates 5000+ unique conversational pairs by assembling components:
- User concerns (life situations)
- Empathy openings 
- Psychology insights (plain language)
- Practical advice
- Real-life examples

Output:
  data_psychology/sft_psych.h5       (train)
  data_psychology/sft_psych_dev.h5   (dev)

Usage:
    python scripts/prepare_psychology_sft.py --out_dir data_psychology --n_target 10000
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import h5py
import numpy as np

from src.post_training.chat_template import encode_chat, get_tokenizer
from src.post_training.sft import pack_examples


# ═══════════════════════════════════════════════════════════════════════
#  COMPONENT POOLS  —  mix & match to create thousands of conversations
# ═══════════════════════════════════════════════════════════════════════

USER_CONCERNS = [
    # (concern_description, is_about_self=True/False)
    "在会议上被领导当众批评了，好几个人都看着，尴尬得要死",
    "发了朋友圈没人点赞，过了一会默默删掉了",
    "跟朋友聊天的时候说错话了，晚上躺床上还在反复回想",
    "喜欢一个人但不敢表白，怕被拒绝连朋友都做不成",
    "面试的时候紧张得话都说不利索，果然没通过",
    "同事们都出去聚餐没叫我，看到他们发的朋友圈很难过",
    "新公司入职一个月了还是融不进去，午饭都是一个人吃",
    "跟男朋友吵架了，明明是他不对但最后哄人的是我",
    "室友天天半夜打游戏，说了好几次都不改",
    "在电梯里遇到领导不知道说什么，全程尴尬沉默",
    "减肥三天就放弃了，觉得自己好没意志力",
    "考试前总是肚子疼，考完就好了",
    "经常做同一个梦，梦到自己在考试但一道题都不会",
    "被家里人催婚催得不想回家",
    "好朋友借钱不还，又不好意思开口要",
    "看到前女友/前男友的新动态，心里堵得慌",
    "工作三年了感觉什么都没学到，想跳槽又不敢",
    "每次想定下心学习，拿起手机就放不下了",
    "在团队里总是干最多活但功劳是别人的",
    "对未来的规划很迷茫，不知道自己喜欢什么",
    "明知道 deadline 要到了就是不想开始做",
    "看到同龄人都买房买车了，自己还在租房",
    "跟父母视频的时候他们总说『你瘦了』，听着心里难受",
    "生病了一个人去打点滴，觉得自己好惨",
    "在社交媒体上看到别人过得很好，突然很焦虑",
    "养了一年的宠物走丢了，哭了好几天",
    "被人背后说坏话，知道了之后不知道该怎么面对",
    "买了件新衣服穿出门，总感觉别人在看我",
    "三十岁了还在做基础岗位，觉得人生没希望了",
    "每次想表达自己的想法，话到嘴边又咽回去了",
    "看到别人升职加薪，嘴上说恭喜心里不是滋味",
    "被最好的朋友拉黑了，不知道发生了什么",
    "第一次去健身房，感觉所有人都在看我这个菜鸟",
    "和合租的人因为卫生问题吵了一架，现在见面很尴尬",
    "辞职了还没找到下家，存款快花完了很焦虑",
    "每次跟父母打电话都能吵起来，但挂了又后悔",
    "在聚会上总是那个默默坐在角落的人",
    "被人说想太多，但我控制不住自己不去想",
    "考研二战又失败了，不知道该不该三战",
    "发现自己的发际线越来越高，焦虑到睡不着",
]

EMPATHY_OPENINGS = [
    "哎，这种感觉我太懂了。",
    "我特别理解你这种感受。",
    "你说的这个情况太真实了，很多人都会遇到。",
    "天哪，换我我也难受。",
    "说实话，这真的太让人心塞了。",
    "我懂，这种感觉就像心里堵了一块石头。",
    "不是你的问题，这种处境谁都会不舒服。",
    "你说的这个我太有共鸣了。",
    "唉，这种事确实挺难处理的。",
    "我跟你说，你这种反应太正常了。",
    "别太自责，这真的不全是你的问题。",
    "这种事放在谁身上都不好受。",
    "我能想象你有多难受。",
    "你说的这个情况，我身边好多人都有类似的经历。",
    "这确实挺让人崩溃的。",
    "听你这么说，我心里也咯噔一下。",
    "说真的，你能说出来就已经很勇敢了。",
    "先给你一个隔空的拥抱。",
    "这年头谁还没点糟心事呢，你一点都不孤单。",
    "我觉得你已经做得很好了，真的。",
    "你信不信，换个人遇到同样的事，未必有你处理得好。",
    "这不就是人生嘛，起起落落的，总会好的。",
    "我特别能体会你说的这种感觉。",
]

PSYCHOLOGY_INSIGHTS = [
    # Each explains a concept in plain language, friend-to-friend
    ("聚光灯效应",
     "我们总觉得所有人都在盯着自己看。但其实每个人最关心的是他们自己，不是我们。"
     "你回想一下，你最近记得谁出过丑吗？大概率想不起来吧。同样的，别人也不会一直记着你的小失误。"),

    ("基本归因错误",
     "我们给自己找理由的时候总是看环境，给别人找原因的时候总看性格。"
     "自己迟到了是因为堵车，别人迟到了就是他不靠谱。其实换位想想，别人可能也有他们的难处。"),

    ("习得性无助",
     "当你反复经历失败之后，大脑就会学会放弃。就像那句话说的——「不是不行，是以为自己不行」。"
     "好消息是，这种状态是可以打破的，从一件特别小的成功开始，重新建立信心。"),

    ("认知失调",
     "当你的行为和你认为的不一致的时候，你的大脑会找各种理由来合理化。"
     "比如你明知道熬夜不好，但还是在刷手机，你的大脑就会说『反正明天可以补觉』。"
     "这不是你意志力差，是大脑在帮你减少心理不适。"),

    ("社会比较",
     "人天生就会跟别人比较，这没办法。但问题是，我们总是拿自己的日常去跟别人的高光时刻比。"
     "你看到的朋友圈都是别人想让你看到的版本——没人会发自己崩溃的那一刻。"
     "你真正的对手不是别人，是昨天的自己。"),

    ("沉没成本陷阱",
     "你投入了一件事，明知道继续下去会损失更大，但舍不得放手。"
     "就像看了一部烂片，钱都花了就想看完。其实最好的选择是及时止损，不管是电影、工作还是感情。"),

    ("确认偏误",
     "我们只想看到支持自己想法的证据，自动忽略相反的。"
     "你觉得同事针对你，就会特别注意他每次没跟你打招呼，但他对别人笑的时候你就当没看见。"
     "意识到这点之后，下次可以试着换个角度看。"),

    ("安慰剂效应",
     "病人吃了没有药效的糖丸，但因为相信这是药，病居然真的好了。"
     "这说明你的大脑比你想象中强大得多——你相信什么，你的身体就会往那个方向走。"
     "所以别小看「信念」的力量。"),

    ("曝光效应",
     "你见一个人或者一个东西的次数越多，你就越容易喜欢它。"
     "这也是为什么新歌多听几遍就觉得好听了，新同事多见几次就顺眼了。"
     "所以对陌生事物的恐惧，很多时候只是「还不够熟悉」而已。"),

    ("自我实现预言",
     "你相信什么，就会不自觉地让什么发生。"
     "你觉得自己今天会搞砸一个汇报，你因为这个想法而紧张，结果真的搞砸了。"
     "反过来也一样——如果你相信自己可以，你的行为就会朝着那个方向走。"),

    ("旁观者效应",
     "在场的人越多，出手帮忙的人反而越少。因为每个人都觉得『总有人会去帮的』。"
     "所以如果你需要帮助，最好是指定某个人，而不是对着一群人喊——打破那种『别人会做』的心理。"),

    ("合理化",
     "弗洛伊德提过一个概念——就是给自己的行为找合理的解释，而不是真实的原因。"
     "就像小朋友打碎了花瓶说『是它自己掉的』——我们大人也在做一样的事，只是更隐蔽了。"
     "对自己诚实一点，其实反而更轻松。"),

    ("焦虑的认知模型",
     "焦虑不是来自事情本身，而是来自你对事情的看法。你的大脑会把危险放大，低估自己的应对能力。"
     "所以下次焦虑的时候，可以问问自己：最坏的情况发生的概率有多大？就算发生了，我真的扛不住吗？"
     "大多数时候，答案会让你轻松很多。"),

    ("心流",
     "你有没有过那种经历——做一件事做到忘记了时间，忘记了饥饿，完全沉浸其中？"
     "这是人最幸福的状态之一。找到一件有挑战但又刚好在你能力范围内的事，你就会很容易进入这个状态。"),

    ("成长型思维",
     "相信能力是可以发展的，而不是固定不变的。"
     "把「我不行」改成「我还不行」，多一个字就完全不一样了。"
     "失败不是对你能力的判决，而是告诉你还有成长的空间。"),

    ("潜意识动机",
     "我们的行为很多是被潜意识驱动的——就是那些你意识不到但真实存在的欲望和恐惧。"
     "那些说不清道不明的直觉、莫名其妙的梦境、脱口而出的口误——都是潜意识在跟你说话。"
     "偶尔停下来听听它在说什么，也许会有意想不到的发现。"),
]

ADVICE_TIPS = [
    "我建议你试试一个小实验——下次觉得自己出丑了，第二天问问身边的人还记得不？答案会让你松一口气。",
    "你可以试一下这个方法：给自己设一个『最小行动』——只做两分钟，两分钟之后想停就停。神奇的是，一旦开始了你就不会想停了。",
    "一个特别管用的小技巧：把「我必须做好」改成「我可以试试看」。这两个说法带来的心理压力完全不一样。",
    "你要不要试一下这个：每次开始焦虑的时候，深呼吸，吸气4秒、憋气4秒、呼气4秒。重复三次，心跳会慢慢降下来。",
    "我自己的方法是：把担心的东西写下来。写出来的那一刻，它就变得没那么可怕了。",
    "你可以试试『五秒法则』——想做一件事的时候倒数54321，数完立刻行动。不给大脑犹豫的时间。",
    "有一个简单的练习：每天睡前想三件今天让你开心的小事，哪怕只是喝到了一杯好咖啡。坚持一段时间，你会发现自己更容易注意到生活中的好事。",
    "下次遇到这种情况，你不用逼自己马上变好。承认『我现在就是很难受』——接受它，它反而会过去得更快。",
    "你试过跟信任的朋友说出来吗？很多时候，把心里的感受说出来，它就减轻了一半。",
    "我给你一个具体的建议：把大目标拆成小到不能再小的步骤。不是『写一份报告』，而是『打开文档，写标题』。完成小目标的成就感会推着你往前走。",
    "你可以试着对自己说：『我已经尽力了，这就够了。』不是所有事情都需要完美，有些事情只需要完成。",
    "下次有这种感受的时候，把手放在胸口，对自己说三句话：『我现在很难受』『每个人都会这样』『我可以对自己好一点』。听起来有点傻，但真的有效。",
    "我建议你设一个『担心时间』——每天下午专门留15分钟用来担心。其他时间脑子里冒出那些念头，就告诉它『下午5点再处理你』。",
    "你可以试一下换个环境。有时候改变的不只是场景，还有你的心情。",
    "如果你的朋友遇到了同样的事，你会怎么安慰他？试着用同样的方式对自己说话。",
    "我记得有个说法是——『你的感受是真实的，但不一定是事实』。你感觉到的和真正发生的，有时候是两回事。",
    "试着把『为什么是我』改成『这教会了我什么』——虽然有点鸡汤，但这个视角转换确实有用。",
    "有时候我们需要的是被理解，而不是被解决。所以你不用急着找到答案，有人听你说就很好。",
    "下次感觉自己要被情绪淹没的时候，试试先停下来喝杯水。这个简单的动作能打断情绪的螺旋上升。",
    "我经常跟自己说的一句话：『这不是世界末日，这只是今天。』明天又是新的一天。",
]

EXAMPLES_POOL = [
    "就像你第一次用新手机的时候觉得不习惯，用了一周就回不去了——大脑需要时间适应新东西。",
    "就像学游泳，你不可能看视频就学会，得下水呛几口水才行。",
    "就像去一家新餐厅，第一口觉得一般，多吃几口反而觉得还不错。",
    "就像运动一样，最难的是换好衣服出门的那一步，一旦开始了反而挺享受。",
    "就像打游戏，新手关卡总是最简单的，但你不从新手关开始就没法打BOSS。",
    "就像你第一次做一道菜，可能不好吃，但第二次、第三次就会越来越好。",
    "就像听一首新歌，第一次觉得一般，第十次已经能跟着哼了。",
    "就像拼图，一开始全是散乱的碎片看不出什么，但拼着拼着轮廓就出来了。",
    "就像走一条没走过的路，一开始不确定方向，走着走着就认路了。",
    "就像你第一次用某个软件觉得好复杂，用了一个月之后闭着眼睛都知道点哪里。",
    "就像你第一次开车上路的时候紧张得手心出汗，开了一个月之后已经可以一边开车一边听歌了。",
    "就像健身一样，前几次去的时候浑身酸疼想放弃，坚持两周之后反而一天不动就不舒服。",
    "就像我们在黑暗的房间里找开关，一开始摸摸索索的，但只要摸到了一次，下次就知道了。",
    "就像学外语一样，一开始听不懂也说不出，但每天听一点，慢慢地就发现能听懂了。",
    "就像一个塞住的洗手池，你得先把堵着的东西清掉，水才能流下去——情绪也是这样的。",
]


# ═══════════════════════════════════════════════════════════════════════
#  GENERATOR
# ═══════════════════════════════════════════════════════════════════════

def generate_pairs(n_target: int, seed: int = 42) -> list[tuple[list[int], list[int]]]:
    """
    Assemble unique conversations from component pools.
    Each conversation = empathy + insight + advice + optional example.
    """
    rng = random.Random(seed)
    examples: list[tuple[list[int], list[int]]] = []
    seen_keys: set[str] = set()

    for attempt in range(n_target * 10):
        if len(examples) >= n_target:
            break

        concern = rng.choice(USER_CONCERNS)
        empathy = rng.choice(EMPATHY_OPENINGS)
        concept_name, insight = rng.choice(PSYCHOLOGY_INSIGHTS)
        advice = rng.choice(ADVICE_TIPS)
        example = rng.choice(EXAMPLES_POOL)
        has_example = rng.random() < 0.4
        has_closing = rng.random() < 0.25

        # Natural ways to introduce a concept — NEVER "你听说过…吗？" (sounds like lecture)
        concept_intros = [
            f"你说的这个，其实跟心理学上说的「{concept_name}」有点像——{insight}",
            f"我想到一个概念叫「{concept_name}」——{insight}",
            f"这不就是「{concept_name}」吗？{insight}",
            f"其实这个在心理学上有个说法叫「{concept_name}」——{insight}",
            None,  # skip naming the concept, just give the insight naturally
            None,
            None,
        ]

        structure_choice = rng.random()

        if structure_choice < 0.25:
            # Short: empathy + insight (no named concept)
            parts = [empathy, insight]
        elif structure_choice < 0.40:
            # Short with concept name
            intro = rng.choice(concept_intros)
            if intro:
                parts = [empathy, intro]
            else:
                parts = [empathy, insight]
        elif structure_choice < 0.60:
            # Medium: empathy + concept + advice
            intro = rng.choice(concept_intros)
            if intro:
                parts = [empathy, intro, advice]
            else:
                parts = [empathy, insight, advice]
        elif structure_choice < 0.80:
            # Full: empathy + concept + example + advice
            intro = rng.choice(concept_intros)
            if intro and has_example:
                parts = [empathy, intro, example, advice]
            elif intro:
                parts = [empathy, intro, advice]
            elif has_example:
                parts = [empathy, insight, example, advice]
            else:
                parts = [empathy, insight, advice]
        else:
            # Full with closing
            closings = [
                "你觉得呢？", "下次可以试试看。", "慢慢来，不着急。",
                "你觉得自己能做到吗？", "希望这个对你有帮助。", "你值得对自己好一点。",
                "你说呢？", "试试看，也许会有不一样的感觉。",
            ]
            intro = rng.choice(concept_intros)
            if intro:
                parts = [empathy, intro, advice, rng.choice(closings)]
            else:
                parts = [empathy, insight, advice, rng.choice(closings)]

        user_text = concern
        assistant_text = "\n\n".join(parts)

        messages = [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": assistant_text},
        ]
        ids, mask = encode_chat(messages)
        # Use assistant content hash for dedup (not first 30 tokens that include template wrapper)
        key = str(assistant_text[:100])
        if key not in seen_keys and len(ids) <= 1024 and sum(mask) > 0:
            seen_keys.add(key)
            examples.append((ids, mask))

    # Shuffle
    rng.shuffle(examples)
    print(f"  generated {len(examples)} unique SFT examples")
    return examples


def write_packed(examples, context_length: int, out_path: str) -> int:
    """Pack examples and write to HDF5."""
    tokens, masks = pack_examples(examples, context_length)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with h5py.File(out_path, "w") as f:
        f.create_dataset("tokens", data=tokens)
        f.create_dataset("loss_mask", data=masks)
    print(f"  wrote {tokens.shape[0]} packed rows x {context_length} -> {out_path}")
    return tokens.shape[0]


def main():
    p = argparse.ArgumentParser(description="Prepare psychology SFT data — friend-style")
    p.add_argument("--out_dir", default="data_psychology")
    p.add_argument("--context_length", type=int, default=1024)
    p.add_argument("--n_target", type=int, default=10000, help="Target number of examples")
    p.add_argument("--dev_frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("=" * 60)
    print("🧠 Master of Freud's LLM — SFT Data (Friend-Style)")
    print("=" * 60)

    examples = generate_pairs(args.n_target, args.seed)
    rng = np.random.default_rng(args.seed)
    rng.shuffle(examples)

    n_dev = max(1, int(len(examples) * args.dev_frac))
    dev, train = examples[:n_dev], examples[n_dev:]

    print(f"\nTotal: {len(examples)} | Train: {len(train)} | Dev: {len(dev)}")

    write_packed(train, args.context_length, os.path.join(args.out_dir, "sft_psych.h5"))
    write_packed(dev, args.context_length, os.path.join(args.out_dir, "sft_psych_dev.h5"))

    # Preview first 3
    preview_path = os.path.join(args.out_dir, "sft_psych_preview.json")
    preview = []
    for ids, mask in train[:3]:
        text = get_tokenizer().decode(ids)
        preview.append({"text": text})
    with open(preview_path, "w", encoding="utf-8") as f:
        json.dump(preview, f, ensure_ascii=False, indent=2)
    print(f"  preview -> {preview_path}")

    total_tokens = sum(len(ids) for ids, _ in examples)
    print(f"\n✅ Done! {len(examples)} examples, {total_tokens:,} total tokens")


if __name__ == "__main__":
    main()
