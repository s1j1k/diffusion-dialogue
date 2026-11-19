# Trial: v0.1: CPU sanity run (7.5M, 2k steps)

Run folder `runs/cpu-small` · recorded 2026-11-19

First trained model after the rewrite (code commit db2613f). Purpose: check that the fixed pipeline learns. Probe shows it denoises well at moderate noise; samples are word soup. Clamping at every step locks in tokens nearest the origin ('traveler'); no clamping is best. Model version v0.1 in models/registry.json.

## Configuration

| Setting | Value |
| --- | --- |
| `tokenizer` | bert-base-uncased |
| `seq_len` | 64 |
| `emb_dim` | 128 |
| `d_model` | 256 |
| `n_layers` | 4 |
| `n_heads` | 4 |
| `dropout` | 0.1 |
| `time_dim` | 128 |
| `num_steps` | 2000 |
| `schedule` | sqrt |
| `nll_fraction` | 0.25 |
| `lr` | 0.0003 |
| `weight_decay` | 0.0 |
| `batch_size` | 32 |
| `train_steps` | 2000 |
| `warmup_steps` | 200 |
| `grad_clip` | 1.0 |
| `ema_decay` | 0.999 |
| `seed` | 0 |
| `train_pairs` | 100000 |
| `params` | 7477632 |

## Training

Final step 1999: train loss 0.2645, wall time 17.4 h (includes any time the machine slept).

| Step | Train loss | Valid loss |
| --- | --- | --- |
| 0 | 1.0755 | 1.0791 |
| 500 | 0.3857 | 0.7964 |
| 1000 | 0.2950 | 0.5266 |
| 1500 | 0.2815 | 0.3939 |
| 1999 | 0.2645 | 0.3279 |

## Denoising probe

Share of response words recovered in one denoising step from noise level t (`scripts/denoise_probe.py`). High accuracy here with poor samples points at the sampler.

| t | 0 | 50 | 200 | 500 | 1000 | 1500 | 1999 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| train | 0.82 | 0.79 | 0.73 | 0.65 | 0.53 | 0.37 | 0.01 |
| test | 0.85 | 0.82 | 0.74 | 0.66 | 0.55 | 0.40 | 0.01 |

Tokens nearest the origin of embedding space: `traveler`, `enclosure`, `sophomore`, `being`, `[unused970]`, `##lity`, `anthropology`, `501`

## Evaluations

| Sampling | Prompts | Corpus BLEU | Sentence BLEU | Distinct-1 | Distinct-2 | Avg length | Time |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 50 steps, clamp all steps | 200 | 0.0003 | 0.0029 | 0.044 | 0.260 | 18.4 | 326 s |
| 50 steps, clamp t≤1000 | 200 | 0.0002 | 0.0059 | 0.088 | 0.585 | 25.8 | 190 s |
| 50 steps, clamp t≤500 | 200 | 0.0001 | 0.0062 | 0.103 | 0.674 | 25.2 | 127 s |
| 50 steps, no clamp | 200 | 0.0002 | 0.0082 | 0.088 | 0.611 | 26.1 | 115 s |

### Samples: 50 steps, clamp all steps

- **Prompt 3:** i wear glasses and at the age of 17 i would have been mortified to be seen in a photo wearing them .
  - Reference: yes, i wore glasses from 1st grade on. you would not know this by looking at photos.
  - Generated: expect, traveler enclosure walker to traveler walker traveler [unused970] enclosure.
- **Prompt 4:** will you be requiring a shake today , sir ?
  - Reference: no! get your nasty hands off me
  - Generated: traveler. traveler traveler they enclosure traveler 501. traveler. traveler traveler traveler traveler being enclosure traveler the walker couple a [unused970] walker being being
- **Prompt 6:** great article . thanks for posting
  - Reference: thanks for reading!
  - Generated: traveler revolves being. enclosure traveler the anthropology walker being being traveler traveler traveler traveler traveler traveler traveler. traveler.. traveler. traveler traveler dick did walker walker
- **Prompt 8:** must really hate utah .
  - Reference: from utah. can confirm feeling.
  - Generated: have traveler walker traveler traveler reward traveler traveler. enclosure traveler reward enclosure. walker walker being anthropology enclosure
- **Prompt 10:** i do n't think jorah ever called her daenerys
  - Reference: that would be because he was her sworn sword and her title was khaleesi. you did n ' t see davos calling stannis by his first name or hound calling joffrey by his.
  - Generated: enclosure. expect walker, being walker traveler traveler being being walker like walker enclosure? enclosure...
- **Prompt 11:** she can warg cats ?
  - Reference: that ' s how she gets her eyes back from the kindly man.
  - Generated: being where traveler enclosure walker. traveler being walker plays walker traveler walker traveler being.. will! being. traveler 501. walker anthropology anthropology. these walker. being

### Samples: 50 steps, clamp t≤1000

- **Prompt 3:** i wear glasses and at the age of 17 i would have been mortified to be seen in a photo wearing them .
  - Reference: yes, i wore glasses from 1st grade on. you would not know this by looking at photos.
  - Generated: did, did enclosureaus to! right human [unused970] enclosure reward beoo anthropology. [unused970] traveler enclosure did human like enclosure will [unused970] traveler likeaus being like ¾.
- **Prompt 4:** will you be requiring a shake today , sir ?
  - Reference: no! get your nasty hands off me
  - Generated: traveler expect traveler traveler they just things 501 like just will they butving but being humanlity the 501 couple a [unused970] ve being enclosure traveler rather 501 maris like did. enclosure walker these walker human. human coupleoo and that have dat but traveler did anthropology 501
- **Prompt 6:** great article . thanks for posting
  - Reference: thanks for reading!
  - Generated: traveler revolves bai ' enclosure traveler the right walker being being enclosure 501 he traveler - like did will will couple will military.lity right was did willlitylity a but?
- **Prompt 8:** must really hate utah .
  - Reference: from utah. can confirm feeling.
  - Generated: have traveler boasted maris traveler t [unused970] anthropology. usually are sophomore things. 501 expect reward like human rightaus. enclosure walker enclosure!.
- **Prompt 10:** i do n't think jorah ever called her daenerys
  - Reference: that would be because he was her sworn sword and her title was khaleesi. you did n ' t see davos calling stannis by his first name or hound calling joffrey by his.
  - Generated: enclosure. walker [unused970], being [unused970] traveler anthropology 501 being walker like wouldlity? if coupleoo andoo. traveler worth
- **Prompt 11:** she can warg cats ?
  - Reference: that ' s how she gets her eyes back from the kindly man.
  - Generated: but where as like!. anthropology being couple plays true traveler about couple like did is for! ¾. there part is human while for.. these maris. worth would and enclosure [unused970] being being

### Samples: 50 steps, clamp t≤500

- **Prompt 3:** i wear glasses and at the age of 17 i would have been mortified to be seen in a photo wearing them .
  - Reference: yes, i wore glasses from 1st grade on. you would not know this by looking at photos.
  - Generated: did, did beaus to! right human [unused970] s reward be did anthropology. [unused970] traveler enclosure did human like enclosure will [unused970] traveler likeaus being like
- **Prompt 4:** will you be requiring a shake today , sir ?
  - Reference: no! get your nasty hands off me
  - Generated: being expect traveler traveler they just have 501 like did will they butving but being human but the 501 mps a have ve likely! traveler rather 501 maris like did. traveler walker 501 being human. maris coupleoo and that have like but traveler did anthropology walker
- **Prompt 6:** great article . thanks for posting
  - Reference: thanks for reading!
  - Generated: traveler revolves bai ' did traveler the right walker being beingaus as he traveler - like did walker will couple will military.lity right was did maris 501 human a but?
- **Prompt 8:** must really hate utah .
  - Reference: from utah. can confirm feeling.
  - Generated: have traveler boasted maris traveler t [unused970]!. dat are sophomore know. 501 walker reward not human rightaus. enclosure enclosure!.
- **Prompt 10:** i do n't think jorah ever called her daenerys
  - Reference: that would be because he was her sworn sword and her title was khaleesi. you did n ' t see davos calling stannis by his first name or hound calling joffrey by his.
  - Generated: your. walker [unused970], being have meetings these 501 being re like wouldlity? if coupleoo andoo. maris worth
- **Prompt 11:** she can warg cats ?
  - Reference: that ' s how she gets her eyes back from the kindly man.
  - Generated: but where as like!. anthropology all are plays true traveler about couple like do is for! be. there part is human while for.. these maris. worth would and enclosure [unused970] being being

### Samples: 50 steps, no clamp

- **Prompt 3:** i wear glasses and at the age of 17 i would have been mortified to be seen in a photo wearing them .
  - Reference: yes, i wore glasses from 1st grade on. you would not know this by looking at photos.
  - Generated: did, did be mikhail to do be human [unused970] s not be did your. [unused970] traveler as did human like like will [unused970] traveler be was
- **Prompt 4:** will you be requiring a shake today , sir ?
  - Reference: no! get your nasty hands off me
  - Generated: beingving like couple they just have 501 like just will they but was but being human but the these boasted a have so just your are just 501 maris like this. 501 right be being as. be coupleoo and like have like but traveler like rather walker
- **Prompt 6:** great article . thanks for posting
  - Reference: thanks for reading!
  - Generated: traveler all that ' consuming did the right be being with walker was he traveler right like do in will but be your are your like was did all so 501 a but? [unused970]
- **Prompt 8:** must really hate utah .
  - Reference: from utah. can confirm feeling.
  - Generated: have traveler where while traveler t ratherzier. was are walker know. 501 walker usually not in rightaus. right enclosure!.
- **Prompt 10:** i do n't think jorah ever called her daenerys
  - Reference: that would be because he was her sworn sword and her title was khaleesi. you did n ' t see davos calling stannis by his first name or hound calling joffrey by his.
  - Generated: right. are [unused970], was have will right 501 for can on wouldlity? if couple these and these. maris though
- **Prompt 11:** she can warg cats ?
  - Reference: that ' s how she gets her eyes back from the kindly man.
  - Generated: but where as like!. moved all are that and 501 these have like do is for! be right there part is so so for like your these be. just s and 501 [unused970] my being
