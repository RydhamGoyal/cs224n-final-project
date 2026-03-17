"""
Compute chrF scores for generated sonnets vs Shakespeare originals.
Compares full fine-tuning vs LoRA sonnet completions.
"""

import re

# The actual Shakespeare sonnets 145-156 (the held-out ones)
# Only the completion part (lines 4+), since lines 1-3 are the prompt
REFERENCE_COMPLETIONS = {
    0: """But when she saw my woeful state,
Straight in her heart did mercy come,
Chiding that tongue that, ever sweet,
Was used in giving gentle doom,
And taught it thus anew to greet:
"I hate" she altered with an end
That followed it as gentle day
Doth follow night, who, like a fiend,
From heaven to hell is flown away.
  "I hate" from hate away she threw,
  And saved my life, saying "not you.\"""",

    1: """Painting thy outward walls so costly gay?
Why so large cost, having so short a lease,
Dost thou upon thy fading mansion spend?
Shall worms, inheritors of this excess,
Eat up thy charge? Is this thy body's end?
Then, soul, live thou upon thy servant's loss
And let that pine to aggravate thy store.
Buy terms divine in selling hours of dross;
Within be fed, without be rich no more.
  So shalt thou feed on death, that feeds on men,
  And death once dead, there's no more dying then.""",

    2: """Th' uncertain sickly appetite to please.
My reason, the physician to my love,
Angry that his prescriptions are not kept,
Hath left me, and I desperate now approve
Desire is death, which physic did except.
Past cure I am, now reason is past care,
And frantic mad with evermore unrest.
My thoughts and my discourse as madmen's are,
At random from the truth vainly expressed,
  For I have sworn thee fair and thought thee bright,
  Who art as black as hell, as dark as night.""",

    3: """That censures falsely what they see aright?
If that be fair whereon my false eyes dote,
What means the world to say it is not so?
If it be not, then love doth well denote
Love's eye is not so true as all men's "No."
How can it? O, how can love's eye be true,
That is so vexed with watching and with tears?
No marvel then though I mistake my view;
The sun itself sees not till heaven clears.
  O cunning love, with tears thou keep'st me blind
  Lest eyes, well seeing, thy foul faults should find!""",

    4: """Am I not shunned by thee when I am sent?
What merit do I in myself respect
That is so proud thy service to despise,
When all my best doth worship thy defect,
Commanded by the motion of thine eyes?
But, love, hate on, for now I know thy mind.
Those that can see thou lov'st, and I am blind.
  O, from what power hast thou this powerful might
  With insufficiency my heart to sway?
  To make me give the lie to my true sight
  And swear that brightness doth not grace the day?""",

    5: """And swear that brightness doth not grace the day?
Whence hast thou this becoming of things ill,
That in the very refuse of thy deeds
There is such strength and warrantise of skill
That in my mind thy worst all best exceeds?
Who taught thee how to make me love thee more,
The more I hear and see just cause of hate?
O, though I love what others do abhor,
With others thou shouldst not abhor my state.
  If thy unworthiness raised love in me,
  More worthy I to be beloved of thee.""",

    6: """Lest guilty of my faults thy sweet self prove.
For, thou betraying me, I do betray
My nobler part to my gross body's treason.
My soul doth tell my body that he may
Triumph in love; flesh stays no farther reason,
But, rising at thy name, doth point out thee
As his triumphant prize. Proud of this pride,
He is contented thy poor drudge to be,
To stand in thy affairs, fall by thy side.
  No want of conscience hold it that I call
  Her "love" for whose dear love I rise and fall.""",

    7: """In vowing new hate after new love bearing.
But why of two oaths' breach do I accuse thee
When I break twenty? I am perjured most,
For all my vows are oaths but to misuse thee,
And all my honest faith in thee is lost,
For I have sworn deep oaths of thy deep kindness,
Oaths of thy love, thy truth, thy constancy,
And to enlighten thee gave eyes to blindness
Or made them swear against the thing they see.
  For I have sworn thee fair. More perjured eye,
  To swear against the truth so foul a lie.""",

    8: """In a cold valley-fountain of that ground,
Which from Love's fire took heat perpetual,
Growing a bath and healthful remedy
For men diseased; but I, my mistress' thrall,
Came there for cure, and this by that I prove:
Love's fire heats water, water cools not love.""",

    9: """Came tripping by; but in her maiden hand
The fairest votary took up that fire
Which many legions of true hearts had warmed,
And so the general of hot desire
Was sleeping by a virgin hand disarmed.
This brand she quenched in a cool well by,
Which from Love's fire took heat perpetual,
Growing a bath and healthful remedy
For men diseased; but I, my mistress' thrall,
Came there for cure, and this by that I prove:
Love's fire heats water, water cools not love.""",

    10: """Unlearned in the world's false subtleties.
Thus vainly thinking that she thinks me young,
Although she knows my days are past the best,
Simply I credit her false-speaking tongue;
On both sides thus is simple truth suppressed.
But wherefore says she not she is unjust?
And wherefore say not I that I am old?
O, love's best habit is in seeming trust,
And age in love loves not to have years told.
  Therefore I lie with her and she with me,
  And in our faults by lies we flattered be.""",

    11: """The worser spirit a woman colored ill.
To win me soon to hell, my female evil
Tempteth my better angel from my side,
And would corrupt my saint to be a devil,
Wooing his purity with her foul pride.
And whether that my angel be turned fiend
Suspect I may, yet not directly tell;
But being both from me, both to each friend,
I guess one angel in another's hell.
  Yet this shall I ne'er know, but live in doubt
  Till my bad angel fire my good one out.""",
}


def parse_generated_sonnets(filepath):
    """Parse generated sonnets file, return dict of {sonnet_id: completion_text}."""
    with open(filepath, 'r') as f:
        text = f.read()

    sonnets = {}
    # split on sonnet numbers
    parts = re.split(r'\n(\d+)\n', text)
    # parts alternates: [header, id, text, id, text, ...]
    for i in range(1, len(parts) - 1, 2):
        sonnet_id = int(parts[i])
        sonnet_text = parts[i + 1].strip()
        # remove the first 3 lines (the prompt) to get just the completion
        lines = sonnet_text.split('\n')
        completion = '\n'.join(lines[3:]).strip()
        sonnets[sonnet_id] = completion

    return sonnets


def main():
    try:
        from sacrebleu.metrics import CHRF
    except ImportError:
        print("Installing sacrebleu...")
        import subprocess
        subprocess.check_call(['pip', 'install', 'sacrebleu'])
        from sacrebleu.metrics import CHRF

    chrf = CHRF()

    # parse both generated files
    full_ft = parse_generated_sonnets('predictions/generated_sonnets.txt')
    lora = parse_generated_sonnets('predictions/generated_sonnets_lora.txt')

    # compute per-sonnet and corpus chrF
    print("=" * 60)
    print("chrF Scores: Full Fine-Tuning vs LoRA (Sonnet Generation)")
    print("=" * 60)

    full_ft_hyps = []
    lora_hyps = []
    refs = []

    for sid in sorted(REFERENCE_COMPLETIONS.keys()):
        ref = REFERENCE_COMPLETIONS[sid]
        refs.append(ref)

        ft_comp = full_ft.get(sid, "")
        lo_comp = lora.get(sid, "")
        full_ft_hyps.append(ft_comp)
        lora_hyps.append(lo_comp)

        ft_score = chrf.corpus_score([ft_comp], [[ref]])
        lo_score = chrf.corpus_score([lo_comp], [[ref]])
        print(f"\nSonnet {sid}:")
        print(f"  Full FT chrF: {ft_score.score:.1f}")
        print(f"  LoRA    chrF: {lo_score.score:.1f}")

    # corpus-level scores
    ft_corpus = chrf.corpus_score(full_ft_hyps, [refs])
    lo_corpus = chrf.corpus_score(lora_hyps, [refs])

    print("\n" + "=" * 60)
    print(f"Corpus chrF - Full Fine-Tuning: {ft_corpus.score:.1f}")
    print(f"Corpus chrF - LoRA (r=4):       {lo_corpus.score:.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
