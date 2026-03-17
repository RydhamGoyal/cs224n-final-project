'''
error analysis for comparing full fine-tuning vs LoRA
on paraphrase detection and sonnet generation

python error_analysis.py
'''

import csv
import re
from sacrebleu.metrics import CHRF


def load_preds(path):
    # reads the prediction csv and returns {id: predicted_token}
    preds = {}
    with open(path) as f:
        next(f)  # skip header line
        for line in f:
            if not line.strip():
                continue
            parts = line.strip().split(',')
            preds[parts[0].strip()] = int(parts[1].strip())
    return preds


def load_dev_data(path):
    # reads the quora dev csv to get sentences + ground truth labels
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            try:
                sid = row['id'].strip()
                data[sid] = {
                    's1': row['sentence1'],
                    's2': row['sentence2'],
                    'label': int(float(row['is_duplicate']))
                }
            except:
                pass
    return data


def word_overlap(s1, s2):
    # computes jaccard-ish overlap between two sentences
    # TODO: maybe try other similarity metrics like edit distance?
    w1 = set(s1.lower().split())
    w2 = set(s2.lower().split())
    if not w1 or not w2:
        return 0
    return len(w1 & w2) / min(len(w1), len(w2))


def paraphrase_analysis():
    print("=" * 60)
    print("PARAPHRASE: Full Fine-Tuning vs LoRA")
    print("=" * 60)

    ft_preds = load_preds('predictions/para-dev-output.csv')
    lora_preds = load_preds('predictions/para-dev-output-lora.csv')
    dev_data = load_dev_data('data/quora-dev.csv')

    # 3919 = "no" (not paraphrase), 8505 = "yes" (is paraphrase)
    tok2label = {3919: 0, 8505: 1}

    ids = set(ft_preds) & set(lora_preds) & set(dev_data)
    print(f"\n{len(ids)} examples")

    ft_right = 0
    lora_right = 0
    both_right = 0
    both_wrong = 0
    ft_wins = []   # ft correct but lora wrong
    lora_wins = [] # lora correct but ft wrong

    for sid in ids:
        gt = dev_data[sid]['label']
        fp = tok2label.get(ft_preds[sid], -1)
        lp = tok2label.get(lora_preds[sid], -1)

        ft_ok = fp == gt
        lora_ok = lp == gt

        ft_right += ft_ok
        lora_right += lora_ok

        if ft_ok and lora_ok:
            both_right += 1
        elif not ft_ok and not lora_ok:
            both_wrong += 1
        elif ft_ok:
            ft_wins.append(sid)
        else:
            lora_wins.append(sid)

    n = len(ids)
    print(f"Full FT acc: {ft_right}/{n} ({100*ft_right/n:.1f}%)")
    print(f"LoRA acc:    {lora_right}/{n} ({100*lora_right/n:.1f}%)")
    print(f"\nBoth correct: {both_right}")
    print(f"Both wrong:   {both_wrong}")
    print(f"FT correct, LoRA wrong: {len(ft_wins)}")
    print(f"LoRA correct, FT wrong: {len(lora_wins)}")

    # break down lora errors into false pos vs false neg
    fn = []  # lora says no but answer is yes
    fp_list = []  # lora says yes but answer is no
    for sid in ft_wins:
        gt = dev_data[sid]['label']
        lp = tok2label.get(lora_preds[sid], -1)
        if gt == 1 and lp == 0:
            fn.append(sid)
        else:
            fp_list.append(sid)

    print(f"\nLoRA false negatives (missed paraphrases): {len(fn)}")
    print(f"LoRA false positives (wrong paraphrases):  {len(fp_list)}")

    # print some examples
    print("\n--- LoRA false negatives (should be paraphrase but said no) ---")
    for sid in fn[:5]:
        d = dev_data[sid]
        print(f"  Q1: {d['s1']}")
        print(f"  Q2: {d['s2']}")
        print()

    print("--- LoRA false positives (not paraphrase but said yes) ---")
    for sid in fp_list[:5]:
        d = dev_data[sid]
        print(f"  Q1: {d['s1']}")
        print(f"  Q2: {d['s2']}")
        print()

    # not sure if this is the best way to check but lets see if
    # lora messes up more on longer sentences
    wrong_lens = []
    right_lens = []
    wrong_overlaps = []
    right_overlaps = []

    for sid in ids:
        gt = dev_data[sid]['label']
        lp = tok2label.get(lora_preds[sid], -1)
        total_words = len(dev_data[sid]['s1'].split()) + len(dev_data[sid]['s2'].split())
        ovlp = word_overlap(dev_data[sid]['s1'], dev_data[sid]['s2'])

        if lp == gt:
            right_lens.append(total_words)
            right_overlaps.append(ovlp)
        else:
            wrong_lens.append(total_words)
            wrong_overlaps.append(ovlp)

    print(f"Avg word count when LoRA correct: {sum(right_lens)/len(right_lens):.1f}")
    print(f"Avg word count when LoRA wrong:   {sum(wrong_lens)/len(wrong_lens):.1f}")
    print(f"Avg word overlap when LoRA correct: {sum(right_overlaps)/len(right_overlaps):.3f}")
    print(f"Avg word overlap when LoRA wrong:   {sum(wrong_overlaps)/len(wrong_overlaps):.3f}")


def read_sonnets(path):
    # split the generated sonnets file into individual sonnets
    with open(path) as f:
        text = f.read()
    sonnets = {}
    chunks = re.split(r'\n(\d+)\n', text)
    for i in range(1, len(chunks) - 1, 2):
        sonnets[int(chunks[i])] = chunks[i+1].strip()
    return sonnets


def sonnet_analysis():
    print("\n\n" + "=" * 60)
    print("SONNETS: Full Fine-Tuning vs LoRA")
    print("=" * 60)

    ft = read_sonnets('predictions/generated_sonnets.txt')
    lora = read_sonnets('predictions/generated_sonnets_lora.txt')

    # structural stuff - line counts, word counts
    print("\nStructure:")
    for sid in sorted(ft.keys()):
        ft_lines = [l for l in ft[sid].split('\n') if l.strip()]
        lo_lines = [l for l in lora[sid].split('\n') if l.strip()]
        print(f"  Sonnet {sid}: FT {len(ft_lines)} lines/{len(ft[sid].split())} words, "
              f"LoRA {len(lo_lines)} lines/{len(lora[sid].split())} words")

    # how much of the output vocab is actually shakespearean
    with open('data/sonnets.txt') as f:
        shakes_words = set(f.read().lower().split())

    ft_words = []
    lora_words = []
    for sid in sorted(ft.keys()):
        ft_words.extend(ft[sid].lower().split())
        lora_words.extend(lora[sid].lower().split())

    ft_vocab = set(ft_words)
    lora_vocab = set(lora_words)

    print(f"\nVocab:")
    print(f"  FT unique words: {len(ft_vocab)}, {100*len(ft_vocab & shakes_words)/len(ft_vocab):.1f}% in Shakespeare")
    print(f"  LoRA unique words: {len(lora_vocab)}, {100*len(lora_vocab & shakes_words)/len(lora_vocab):.1f}% in Shakespeare")

    # quick check for modern words that definitely shouldnt show up in shakespeare
    bad_words = ['http', 'www', '.com', 'click', 'post', 'blog', 'email',
                 'university', 'stage', 'intro', 'action', 'serial',
                 'instagram', 'amazon', 'ebay', 'hotel', 'tv']
    ft_bad = sum(1 for w in ft_words if any(b in w.lower() for b in bad_words))
    lora_bad = sum(1 for w in lora_words if any(b in w.lower() for b in bad_words))
    print(f"\nModern/web words: FT={ft_bad}, LoRA={lora_bad}")

    # avg words per line - sonnets should be around 10
    ft_wpl = [len(l.split()) for s in ft.values() for l in s.split('\n') if l.strip()]
    lo_wpl = [len(l.split()) for s in lora.values() for l in s.split('\n') if l.strip()]
    print(f"Avg words/line: FT={sum(ft_wpl)/len(ft_wpl):.1f}, LoRA={sum(lo_wpl)/len(lo_wpl):.1f}")

    # chrF against real shakespeare completions
    # these are the actual continuations of sonnets 145-156
    refs = {
        0: "But when she saw my woeful state,\nStraight in her heart did mercy come,\nChiding that tongue that, ever sweet,\nWas used in giving gentle doom,\nAnd taught it thus anew to greet:\n\"I hate\" she altered with an end\nThat followed it as gentle day\nDoth follow night, who, like a fiend,\nFrom heaven to hell is flown away.\n  \"I hate\" from hate away she threw,\n  And saved my life, saying \"not you.\"",
        1: "Painting thy outward walls so costly gay?\nWhy so large cost, having so short a lease,\nDost thou upon thy fading mansion spend?\nShall worms, inheritors of this excess,\nEat up thy charge? Is this thy body's end?\nThen, soul, live thou upon thy servant's loss\nAnd let that pine to aggravate thy store.\nBuy terms divine in selling hours of dross;\nWithin be fed, without be rich no more.\n  So shalt thou feed on death, that feeds on men,\n  And death once dead, there's no more dying then.",
        2: "Th' uncertain sickly appetite to please.\nMy reason, the physician to my love,\nAngry that his prescriptions are not kept,\nHath left me, and I desperate now approve\nDesire is death, which physic did except.\nPast cure I am, now reason is past care,\nAnd frantic mad with evermore unrest.\nMy thoughts and my discourse as madmen's are,\nAt random from the truth vainly expressed,\n  For I have sworn thee fair and thought thee bright,\n  Who art as black as hell, as dark as night.",
        3: "That censures falsely what they see aright?\nIf that be fair whereon my false eyes dote,\nWhat means the world to say it is not so?\nIf it be not, then love doth well denote\nLove's eye is not so true as all men's \"No.\"\nHow can it? O, how can love's eye be true,\nThat is so vexed with watching and with tears?\nNo marvel then though I mistake my view;\nThe sun itself sees not till heaven clears.\n  O cunning love, with tears thou keep'st me blind\n  Lest eyes, well seeing, thy foul faults should find!",
        4: "Am I not shunned by thee when I am sent?\nWhat merit do I in myself respect\nThat is so proud thy service to despise,\nWhen all my best doth worship thy defect,\nCommanded by the motion of thine eyes?\nBut, love, hate on, for now I know thy mind.\nThose that can see thou lov'st, and I am blind.",
        5: "And swear that brightness doth not grace the day?\nWhence hast thou this becoming of things ill,\nThat in the very refuse of thy deeds\nThere is such strength and warrantise of skill\nThat in my mind thy worst all best exceeds?\nWho taught thee how to make me love thee more,\nThe more I hear and see just cause of hate?\nO, though I love what others do abhor,\nWith others thou shouldst not abhor my state.\n  If thy unworthiness raised love in me,\n  More worthy I to be beloved of thee.",
        6: "Lest guilty of my faults thy sweet self prove.\nFor, thou betraying me, I do betray\nMy nobler part to my gross body's treason.\nMy soul doth tell my body that he may\nTriumph in love; flesh stays no farther reason,\nBut, rising at thy name, doth point out thee\nAs his triumphant prize. Proud of this pride,\nHe is contented thy poor drudge to be,\nTo stand in thy affairs, fall by thy side.\n  No want of conscience hold it that I call\n  Her \"love\" for whose dear love I rise and fall.",
        7: "In vowing new hate after new love bearing.\nBut why of two oaths' breach do I accuse thee\nWhen I break twenty? I am perjured most,\nFor all my vows are oaths but to misuse thee,\nAnd all my honest faith in thee is lost,\nFor I have sworn deep oaths of thy deep kindness,\nOaths of thy love, thy truth, thy constancy,\nAnd to enlighten thee gave eyes to blindness\nOr made them swear against the thing they see.\n  For I have sworn thee fair. More perjured eye,\n  To swear against the truth so foul a lie.",
        8: "In a cold valley-fountain of that ground,\nWhich from Love's fire took heat perpetual,\nGrowing a bath and healthful remedy\nFor men diseased; but I, my mistress' thrall,\nCame there for cure, and this by that I prove:\nLove's fire heats water, water cools not love.",
        9: "Came tripping by; but in her maiden hand\nThe fairest votary took up that fire\nWhich many legions of true hearts had warmed,\nAnd so the general of hot desire\nWas sleeping by a virgin hand disarmed.\nThis brand she quenched in a cool well by,\nWhich from Love's fire took heat perpetual,\nGrowing a bath and healthful remedy\nFor men diseased; but I, my mistress' thrall,\nCame there for cure, and this by that I prove:\nLove's fire heats water, water cools not love.",
        10: "Unlearned in the world's false subtleties.\nThus vainly thinking that she thinks me young,\nAlthough she knows my days are past the best,\nSimply I credit her false-speaking tongue;\nOn both sides thus is simple truth suppressed.\nBut wherefore says she not she is unjust?\nAnd wherefore say not I that I am old?\nO, love's best habit is in seeming trust,\nAnd age in love loves not to have years told.\n  Therefore I lie with her and she with me,\n  And in our faults by lies we flattered be.",
        11: "The worser spirit a woman colored ill.\nTo win me soon to hell, my female evil\nTempteth my better angel from my side,\nAnd would corrupt my saint to be a devil,\nWooing his purity with her foul pride.\nAnd whether that my angel be turned fiend\nSuspect I may, yet not directly tell;\nBut being both from me, both to each friend,\nI guess one angel in another's hell.\n  Yet this shall I ne'er know, but live in doubt\n  Till my bad angel fire my good one out.",
    }

    chrf = CHRF()
    ft_hyps = []
    lora_hyps = []
    ref_list = []

    print("\nchrF scores:")
    for sid in sorted(refs.keys()):
        ref_list.append(refs[sid])
        # skip prompt (first 3 lines) to just get the completion
        ft_comp = '\n'.join(ft[sid].split('\n')[3:]).strip() if sid in ft else ""
        lo_comp = '\n'.join(lora[sid].split('\n')[3:]).strip() if sid in lora else ""
        ft_hyps.append(ft_comp)
        lora_hyps.append(lo_comp)

        fs = chrf.corpus_score([ft_comp], [[refs[sid]]])
        ls = chrf.corpus_score([lo_comp], [[refs[sid]]])
        print(f"  Sonnet {sid}: FT={fs.score:.1f}, LoRA={ls.score:.1f}")

    ft_total = chrf.corpus_score(ft_hyps, [ref_list])
    lo_total = chrf.corpus_score(lora_hyps, [ref_list])
    print(f"\nCorpus chrF: Full FT={ft_total.score:.1f}, LoRA={lo_total.score:.1f}")


if __name__ == "__main__":
    paraphrase_analysis()
    sonnet_analysis()
