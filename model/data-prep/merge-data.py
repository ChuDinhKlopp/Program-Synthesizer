import json

with open("data_formal_2.json") as df:
    data_formal = json.load(df)

with open("data_casual_2.json") as dc:
    data_casual = json.load(dc)

with open("data_informal_2.json") as di:
    data_informal = json.load(di)

with open("data_without_text_2.json") as dw:
	data_w = json.load(dw)
	
with open("manimml-data-3k.json") as mn:
	data_manim_3k = json.load(mn)

[data_formal.extend(d) for d in (data_casual, data_informal, data_w, data_manim_3k)]

json_object = json.dumps(data_formal, indent=4)

with open('merge.json', 'w') as m:
    m.write(json_object)
