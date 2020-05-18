#!/usr/bin/env bash
relations=(
    "sports@sports_team@sport"
    "tv@tv_program@languages"
    "location@capital_of_administrative_division@capital_of.@location@administrative_division_capital_relationship@administrative_division"
    "organization@organization_founder@organizations_founded"
     "music@artist@origin"
#     "film@director@film"
#     "film@film@written_by"
#      "people@person@place_of_birth"
#     "people@person@nationality"
#    "film@film@language"
    )
#relations=("@tv@tv_program@languages")

for relation in ${relations[*]}; do
    python3 filter_train.py  $relation
done
