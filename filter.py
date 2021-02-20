import os
import json

HO3D_ROOT = '/Disk1/guyi/HO3D_guyi'

ret = []

last_len = len(ret)
for d in sorted(os.listdir(HO3D_ROOT)):
    # if 'MC' in d:
        # continue
        # pass
    mapping_file = os.path.join(HO3D_ROOT, d, 'mapping3.json')
    if not os.path.isfile(mapping_file):
        continue
    with open(mapping_file) as f:
        js = json.loads(f.read())
    for hand_idx, mapping in js.items():
        hand_idx = int(hand_idx)
        obj = mapping['obj_index']
        grasp = mapping['grasp_index']
        five_dis = mapping['five_dis']
        mapping_dis = mapping['mapping_dis']
        if five_dis > 0.2 or mapping_dis > 0.03:
            # pass
            continue
        # if len(ret) and ret[-1][0] == d and ret[-1][3] == grasp:
            # if mapping_dis < ret[-1][-1]:
                # ret[-1] = d, hand_idx, obj, grasp, five_dis, mapping_dis
            # continue
        ret.append((d, hand_idx, obj, grasp, five_dis, mapping_dis))
    print(d, len(ret) - last_len)
    last_len = len(ret)

with open('filtered.csv', 'w') as csv:
    for d, h, o, g, f, m in ret:
        csv.write(f'{d},{h},{o},{g},{f},{m}\n')
with open('filtered.json', 'w') as j:
    j.write(json.dumps(ret))