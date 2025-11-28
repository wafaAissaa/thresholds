import requests
from urllib.parse import quote_plus
import json

services = "dependency_relations" # "tense_features" # "morphology_features" "new_family"

# message = "Rends-moi ce stylo !"
# message = "La commune est la ville ou le village où vous habitez." # ou:ville,village
# message = "On peut discuter si tu veux, je pense que ça fait du bien."
# message = "Il pouvait s'agir de propriétaires ou de métayers, mais pratiquement jamais de domestiques mariés, car, pour convoler, il était nécessaire de disposer d'un lopin et d'une demeure."
message = "« Sortir du syndrome Tarzan »    La brigade anti-négrophobie est un collectif antiraciste. Il existe depuis 2005. Pourquoi ce collectif ? Quelles sont ses actions? Explications avec Franco, le porte-parole du collectif.    Qu'est-ce que la négrophobie?  Tout ce qui est en rapport avec la peur, la haine, le mépris, le rejet des personnes et des cultures noires.    Comment s'est formée la brigade anti-négrophobie?   La brigade est née en 2005. Des incendies ont eu lieu en région parisienne. Ils ont révélé des discriminations raciales. Puis il y a eu la mort de Zyed et Bouna. Cela a posé la question du contrôle au faciès. Pourquoi les Noirs et les Arabes sont plus touchés par le contrôle au faciès ? La France veut combattre le racisme mais elle ne s'en donne pas les moyens.    Quelles ont été vos actions?  En 2005, on a occupé le plateau de Canal+. On a demandé la démission de l'animateur Marc Olivier Fogiel. Il avait été reconnu coupable d'incitation à la haine raciale. On s'est aussi enchaîné à l'Assemblée Nationale. La statue de Colbert est devant la Maison du Peuple. Colbert est le père du Code noir (1685) qui inscrivait que les noirs ne sont pas des êtres humains!    Quel a été votre parcours?  Petit, je ne recevais que des images négatives : le syndrome Tarzan. Quand un Blanc et un Noir regardent Tarzan, les deux s'identifient. Pourtant, l'un voudra se défriser les cheveux. Puis, j'ai appris que des Noirs s'étaient battus pour de grandes idées. Je suis travailleur social. Éduquer les mentalités est fondamental dans la lutte contre le racisme et la négrophobie.     Pour en savoir plus : WWW.BRIGADEANTINEGROPHOBIE.COM"


def print_phenomenon(r, phenomenon):
    output_dict = json.loads(r.text)
    for s in output_dict.get("sentences", []):
        sentence = output_dict["sentences"][s]
        for w in sentence.get("words", []):
            word = sentence["words"][w]
            wordp = word.get(phenomenon)
            print(word["text"], wordp)


difficulty_level = "A1"
message_json = json.dumps(message, ensure_ascii=False)


# --- /!\ I realize that the difficulty level is not used in the phenomena server /!\ ---


#server_ip = "0.0.0.0"
server_ip = "192.168.249.77"
port = 8080
r = requests.post(url="http://" + server_ip + ":" + str(port) + "/process_phenomena",data={"raw_text": message_json,
                        "difficulty_level": difficulty_level})
                        # "services": services})


# save the output to a json file
with open("output_phenomena.json", "w") as f:
    f.write(r.text)


# all_phenomena = ["passive", "coordination", "subordination", "clitic_pronouns"]

print("r.text:", r.text)
# r.text = {"sentences": [{"words": [{"word": "\"", "id": 1, "passive": "O", "subordination": "O"}, {"word": "Je", "id": 2, "passive": "O", "subordination": "O"}, {"word": "te", "id": 3, "passive": "O", "subordination": "O"}, {"word": "prête", "id": 4, "passive": "O", "subordination": "O"}, {"word": "ma", "id": 5, "passive": "O", "subordination": "O"}, {"word": "voiture", "id": 6, "passive": "O", "subordination": "O"}, {"word": "à", "id": 7, "passive": "O", "subordination": "O"}, {"word": "condition", "id": 8, "passive": "O", "subordination": "O"}, {"word": "que", "id": 9, "passive": "O", "subordination": "O"}, {"word": "tu", "id": 10, "passive": "O", "subordination": "O"}, {"word": "fasses", "id": 11, "passive": "O", "subordination": "O"}, {"word": "très", "id": 12, "passive": "O", "subordination": "O"}, {"word": "attention", "id": 13, "passive": "O", "subordination": "O"}, {"word": ".", "id": 14, "passive": "O", "subordination": "O"}, {"word": "\"", "id": 15, "passive": "O", "subordination": "O"}]}]}
print_phenomenon(r, "coordination")
