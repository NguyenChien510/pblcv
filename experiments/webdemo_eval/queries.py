"""
Hand-labelled queries for the web demo on two real videos: TESTVIDEO.mp4 (campus, 18 tracks) and ok.mp4 (crowded
station, 149 tracks). Track ids are those of `SearchEngine.analyze` on the video's proxy (webapp/video_io.make_proxy),
max_frames 0, stride 1, conf 0.35; they stay valid while tracking is unchanged (identical boxes were checked up to
TRACKER_VERSION botsort-reid-v8-par). A query is right when any of its ids ranks first (several ids = the same person
split over tracks).
- GT: 9 + 15 general descriptions, written 2026-11-15.
- EXT: 46 station queries written 2026-11-28 from per-track crop sheets before any search, focused on accessories
  (skirt/dress, shorts, backpack, bag, suitcase, hat) and 8 directions checked on drawn trajectories. Weights were
  chosen on EXT[0::2] (with TVPReid val) and checked on EXT[1::2].
"""
GT = {
 "TESTVIDEO": [("A man in a black t-shirt and plaid shorts wearing a cap", [7, 11]),
      ("A man wearing a white t-shirt and black shorts with a black backpack", [8]),
      ("A person with a white and pink backpack", [6, 12]), ("A man in a light blue shirt and dark pants", [4]),
      ("A woman in a blue shirt carrying a black backpack", [2, 15]), ("A man dressed all in black walking", [13, 5]),
      ("A man carrying a plaid shirt and a backpack, wearing black shorts and a cap", [3]),
      ("A person wearing a green top with a grey backpack", [1, 18]), ("A woman wearing khaki pants and a dark blue top", [15])],
 "ok": [("A man in an orange high-visibility vest", [118]), ("A woman in a black dress with red flowers holding a pink bag", [2]),
 ("A woman wearing a navy crop top and light blue shorts", [67, 112]), ("A man in a white shirt with a black backpack and brown shorts wearing a cap", [7]),
 ("A person in an orange shirt and yellow shorts", [14]), ("A woman with a black top and red pants", [22, 135]),
 ("A man in a purple shirt", [47, 132, 116, 90, 98]), ("A man in a blue shirt with a backpack and grey shorts wearing headphones", [61, 140]),
 ("A woman carrying a bag with a British flag and red pants", [63]), ("A person with pink hair and a red backpack", [96]),
 ("A man in a red t-shirt and black shorts with red socks", [11, 91]), ("A woman in a striped dress with a red bag and a sun hat", [74]),
 ("A man wearing a plaid shirt and khaki pants", [145]), ("A man in a turquoise shirt with a black backpack", [49]),
 ("A man in a white polo shirt and beige cargo shorts with a backpack", [149])]}

EXT = [
 ("A woman with a white backpack and a pink top", [3]),
 ("A man in a plaid shirt and black shorts", [8]),
 ("An older woman with grey hair carrying a beige shoulder bag and wearing grey cropped trousers", [10]),
 ("A woman in a sleeveless floral top and black cropped trousers", [12]),
 ("A woman wearing a grey pinafore dress over a white t-shirt", [15]),
 ("A woman in a white top with a purple patterned cardigan, jeans and a crossbody bag", [17]),
 ("A blonde woman in a white and black patterned top and jeans carrying a dark handbag", [18]),
 ("A young woman in denim shorts and a light blue top", [19]),
 ("A man with a long brown patterned scarf over a black top", [21]),
 ("A man in a red t-shirt with glasses", [23, 142]),
 ("A woman in a blue top with a grey backpack", [25]),
 ("A woman in a floral blouse with a black shoulder bag", [29]),
 ("A woman with a large dark green hiking backpack and black cropped trousers", [30]),
 ("A man in an orange t-shirt with a black backpack and black trousers", [33]),
 ("A woman in a black crop top and wide striped orange and white trousers", [34]),
 ("A woman in a white striped t-shirt with a black shoulder bag", [35]),
 ("A man in a white t-shirt and denim shorts carrying a beige shoulder bag", [36]),
 ("A woman in an off-the-shoulder white top and grey wide trousers", [39, 138]),
 ("A woman in a black sleeveless top and denim shorts with a backpack", [40]),
 ("A blonde woman in a long navy patterned maxi dress", [42]),
 ("A man in a navy blue shirt and white shorts", [43]),
 ("A woman in a blue t-shirt and grey jeans pulling a black suitcase", [44]),
 ("A woman in a short navy dress and red shoes", [45]),
 ("A woman in a white sun hat, grey top and white trousers", [46, 108, 85]),
 ("A bald man in a purple shirt with a black backpack", [47]),
 ("A woman in a yellow and white floral skirt", [48]),
 ("A young woman in a colourful top with denim shorts, brown boots and a backpack", [52]),
 ("A woman in a blue denim skirt carrying a green bag", [65]),
 ("A woman in a black t-shirt and jeans with a large beige shoulder bag", [70, 131]),
 ("A woman in a long grey striped dress", [71]),
 ("A woman in a light grey skirt with a suitcase", [73]),
 ("A woman in a white tank top and jeans pulling a blue suitcase", [77]),
 ("A man in a light blue shirt carrying a white plastic bag", [78]),
 ("A man in a black t-shirt pulling two suitcases", [134]),
 ("An old woman with white hair pulling a blue suitcase", [100]),
 ("A woman with an orange backpack and a blue skirt", [99]),
 ("A man in a pink t-shirt and yellow shorts", [110]),
 ("A young man in a black t-shirt and black shorts with a black backpack", [92]),
 # direction (checked on drawn trajectories)
 ("A bald man in a maroon checked shirt walking to the right", [80]),
 ("A woman with curly ginger hair in a white jacket walking to the right", [122]),
 ("A woman in a black t-shirt carrying a large beige shoulder bag walking to the left", [70, 131]),
 ("An older woman with a beige shoulder bag walking to the left", [10]),
 ("A woman in a long dark floral dress walking away from the camera", [55]),
 ("A young woman in a black t-shirt and denim shorts walking away from the camera", [88]),
 ("A woman in a grey t-shirt and a floral skirt walking towards the camera", [107]),
 ("An old man in a white shirt walking towards the camera", [139]),
]

