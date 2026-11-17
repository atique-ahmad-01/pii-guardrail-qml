"""50 hand-written SYNTHETIC examples where a person is identifiable without a direct identifier.

Every person, place, club and event below is invented. Each example is (text, phrase): ``phrase``
is the exact substring that, taken together, would let a reader single out one person. It is
labelled ``CONTEXT_ID``. None of the 28 training labels cover this, so the evaluation scores it
type-agnostically (did any mask cover the phrase?).

STATUS: written by the assistant, awaiting review by the project owner.
"""

from __future__ import annotations

from typing import Any

LABEL = "CONTEXT_ID"

EXAMPLES: list[tuple[str, str]] = [
    (
        "Please send the invoice to the only pediatric surgeon at Lakeview Clinic who also coaches the under-12 chess team.",
        "the only pediatric surgeon at Lakeview Clinic who also coaches the under-12 chess team",
    ),
    (
        "My neighbor, the retired lighthouse keeper on Cobble Lane who lost his left thumb in 1987, keeps complaining about the noise.",
        "the retired lighthouse keeper on Cobble Lane who lost his left thumb in 1987",
    ),
    (
        "Our new hire is the woman who won the 2019 Tasmanian state marathon and now leads the data team in Hobart.",
        "the woman who won the 2019 Tasmanian state marathon and now leads the data team in Hobart",
    ),
    (
        "The patient in bed 4, a 91-year-old former mayor of Ostrava, asked for his medication to be delayed.",
        "a 91-year-old former mayor of Ostrava",
    ),
    (
        "Ask the head librarian of the Riverside branch, the one who published a poetry book about beekeeping last spring.",
        "the head librarian of the Riverside branch, the one who published a poetry book about beekeeping last spring",
    ),
    (
        "The defendant is the only owner of a vintage bakery on Rue Saint-Michel whose twin brother is a famous cellist.",
        "the only owner of a vintage bakery on Rue Saint-Michel whose twin brother is a famous cellist",
    ),
    (
        "Her landlord is the man with the green parrot who runs the hardware store opposite the Kaiserplatz tram stop.",
        "the man with the green parrot who runs the hardware store opposite the Kaiserplatz tram stop",
    ),
    (
        "Call the night-shift nurse on ward C who was featured in last month's newspaper for adopting three rescued greyhounds.",
        "the night-shift nurse on ward C who was featured in last month's newspaper for adopting three rescued greyhounds",
    ),
    (
        "The student who scored the highest in the 2023 regional physics olympiad at Greenfield High is requesting an extension.",
        "The student who scored the highest in the 2023 regional physics olympiad at Greenfield High",
    ),
    (
        "I think the complaint came from the tall man in apartment 12B who plays the accordion every Sunday morning.",
        "the tall man in apartment 12B who plays the accordion every Sunday morning",
    ),
    (
        "He is the sole survivor of the 2004 ferry accident near Split and now teaches sailing in Zadar.",
        "the sole survivor of the 2004 ferry accident near Split and now teaches sailing in Zadar",
    ),
    (
        "Please update the file of the councillor who represents the northern ward and was born on a boat in the Adriatic.",
        "the councillor who represents the northern ward and was born on a boat in the Adriatic",
    ),
    (
        "The applicant is the only left-handed goalkeeper to have played for FC Linden in the last decade.",
        "the only left-handed goalkeeper to have played for FC Linden in the last decade",
    ),
    (
        "Our CEO's husband, who owns the vineyard next to the old mill in Trento, will attend the dinner.",
        "Our CEO's husband, who owns the vineyard next to the old mill in Trento",
    ),
    (
        "The reporter who broke the 2021 water-contamination story in Dunmore is asking for a quote.",
        "The reporter who broke the 2021 water-contamination story in Dunmore",
    ),
    (
        "The teacher of room 5 at Oak Hill Primary, who is also the town's volunteer fire chief, called in sick.",
        "The teacher of room 5 at Oak Hill Primary, who is also the town's volunteer fire chief",
    ),
    (
        "She is the only female taxi driver in Zermatt and is known for her red vintage Mercedes.",
        "the only female taxi driver in Zermatt and is known for her red vintage Mercedes",
    ),
    (
        "Please remind the guy who sits by the window in the third-floor lab and has a prosthetic left hand to sign the form.",
        "the guy who sits by the window in the third-floor lab and has a prosthetic left hand",
    ),
    (
        "The donor wishes to remain anonymous, but she is the 94-year-old widow who funded the new wing of the Marlow Hospital.",
        "the 94-year-old widow who funded the new wing of the Marlow Hospital",
    ),
    (
        "The suspect is described as the identical twin of the pastor of St. Brendan's in Galway.",
        "the identical twin of the pastor of St. Brendan's in Galway",
    ),
    (
        "Our tenant is the Olympic fencer from Gdansk who now runs the yoga studio above the post office.",
        "the Olympic fencer from Gdansk who now runs the yoga studio above the post office",
    ),
    (
        "The witness, a bus driver on route 47 who has a long white beard and a missing front tooth, gave a statement.",
        "a bus driver on route 47 who has a long white beard and a missing front tooth",
    ),
    (
        "I'm writing about the only dentist in Valdemarsvik who also serves as the local mayor.",
        "the only dentist in Valdemarsvik who also serves as the local mayor",
    ),
    (
        "Tell the girl who won last year's spelling bee in Fairview and lives above the flower shop that her prize arrived.",
        "the girl who won last year's spelling bee in Fairview and lives above the flower shop",
    ),
    (
        "Our patient is the lead singer of the band that played the 2022 Sziget opening night and lives in Gyor.",
        "the lead singer of the band that played the 2022 Sziget opening night and lives in Gyor",
    ),
    (
        "The new principal is the man who used to be the youngest chess grandmaster in Norway.",
        "the man who used to be the youngest chess grandmaster in Norway",
    ),
    (
        "The claimant is the florist from Maple Street whose shop burned down during the 2018 carnival.",
        "the florist from Maple Street whose shop burned down during the 2018 carnival",
    ),
    (
        "She is the third-grade teacher whose husband was the pilot of the rescue helicopter at the 2016 Tatra flood.",
        "the third-grade teacher whose husband was the pilot of the rescue helicopter at the 2016 Tatra flood",
    ),
    (
        "The cardiologist who treated the president of Lakeland's chamber of commerce last winter will review the case.",
        "The cardiologist who treated the president of Lakeland's chamber of commerce last winter",
    ),
    (
        "Please ask the intern who speaks fluent Basque and arrived from Bilbao on a bicycle last week to attend.",
        "the intern who speaks fluent Basque and arrived from Bilbao on a bicycle last week",
    ),
    (
        "The elderly gentleman at table 9, the retired judge of the Tallinn district court, wants the window seat.",
        "the retired judge of the Tallinn district court",
    ),
    (
        "The person who reported the leak is the only vegan butcher in the old town of Porto.",
        "the only vegan butcher in the old town of Porto",
    ),
    (
        "Our driver tonight is the woman who finished second in the 2020 Alpine rally and owns the garage on Hauptstrasse.",
        "the woman who finished second in the 2020 Alpine rally and owns the garage on Hauptstrasse",
    ),
    (
        "The author of the anonymous letter is probably the town clerk's brother, the one who restores clocks in Kilkenny.",
        "the town clerk's brother, the one who restores clocks in Kilkenny",
    ),
    (
        "Admit the boy from the blue house at the end of Birch Road whose father is the school's caretaker.",
        "the boy from the blue house at the end of Birch Road whose father is the school's caretaker",
    ),
    (
        "The pharmacist at the corner shop in Bergamo who was born deaf and teaches sign language on Saturdays has agreed to speak.",
        "The pharmacist at the corner shop in Bergamo who was born deaf and teaches sign language on Saturdays",
    ),
    (
        "The patient is the only person in the village of Hallin with a pacemaker and a pet llama.",
        "the only person in the village of Hallin with a pacemaker and a pet llama",
    ),
    (
        "Reach out to the architect who designed the glass footbridge in Rijeka and lives alone with two huskies.",
        "the architect who designed the glass footbridge in Rijeka and lives alone with two huskies",
    ),
    (
        "The new manager is the former goalkeeper whose penalty save decided the 2015 Gothenburg cup final.",
        "the former goalkeeper whose penalty save decided the 2015 Gothenburg cup final",
    ),
    (
        "The complainant is the woman at number 7 whose son was the youngest ever winner of the Krakow piano competition.",
        "the woman at number 7 whose son was the youngest ever winner of the Krakow piano competition",
    ),
    (
        "Please notify the volunteer who drives the mobile library van through the Ardennes villages every Thursday.",
        "the volunteer who drives the mobile library van through the Ardennes villages every Thursday",
    ),
    (
        "The anonymous donor is the retired bus mechanic from Lund who inherited the red barn on the hill.",
        "the retired bus mechanic from Lund who inherited the red barn on the hill",
    ),
    (
        "Her therapist is the psychologist from the Rosenthal practice who has the same name as the city's tallest bridge.",
        "the psychologist from the Rosenthal practice who has the same name as the city's tallest bridge",
    ),
    (
        "The man who found the 1,200-year-old coin hoard near Vik and now runs the souvenir shop there is our guest.",
        "The man who found the 1,200-year-old coin hoard near Vik and now runs the souvenir shop there",
    ),
    (
        "The only paramedic in the valley who is also a licensed beekeeper responded to the call.",
        "The only paramedic in the valley who is also a licensed beekeeper",
    ),
    (
        "She is the district's youngest elected sheriff, elected at 26 in the county north of Cedar Falls.",
        "the district's youngest elected sheriff, elected at 26 in the county north of Cedar Falls",
    ),
    (
        "The man in the grey coat on the 8:15 train to Utrecht, the one who always carries a cello case, left his wallet.",
        "The man in the grey coat on the 8:15 train to Utrecht, the one who always carries a cello case",
    ),
    (
        "Please address the letter to the owner of the small bookstore in Girona who also translates Icelandic sagas.",
        "the owner of the small bookstore in Girona who also translates Icelandic sagas",
    ),
    (
        "The family doctor in Koszalin who delivered the triplets born during the 2010 blizzard has retired.",
        "The family doctor in Koszalin who delivered the triplets born during the 2010 blizzard",
    ),
    (
        "Notify the only translator at the ministry who is fluent in both Maltese and Finnish.",
        "the only translator at the ministry who is fluent in both Maltese and Finnish",
    ),
]


def build_context_docs() -> list[dict[str, Any]]:
    """Turn :data:`EXAMPLES` into documents with one ``CONTEXT_ID`` span each."""
    docs = []
    for i, (text, phrase) in enumerate(EXAMPLES, 1):
        if text.count(phrase) != 1:
            raise ValueError(f"example {i}: phrase must occur exactly once")
        s = text.index(phrase)
        docs.append(
            {
                "id": f"ctx_{i:03d}",
                "language": "English",
                "text": text,
                "spans": [{"start": s, "end": s + len(phrase), "label": LABEL, "value": phrase}],
                "transform": "context_only_synthetic",
                "source_id": f"ctx_{i:03d}",
                "eval_labels": [LABEL],
            }
        )
    return docs
