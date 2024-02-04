import copy
import os
from collections import Counter

import numpy as np
from overcooked_ai_py.mdp.actions import Action, Direction
from overcooked_ai_py.mdp.overcooked_mdp import (ObjectState,
                                                 OvercookedGridworld,
                                                 OvercookedState, Recipe,
                                                 SoupState)


class OvercookedGridworld_(OvercookedGridworld):
    def __init__(
        self,
        terrain,
        start_player_positions,
        start_bonus_orders=[],
        rew_shaping_params=None,
        layout_name="unnamed_layout",
        start_all_orders=[],
        num_items_for_soup=3,
        order_bonus=2,
        start_state=None,
        observation_ranges=None,
        **kwargs,
    ):
        super().__init__(
            terrain,
            start_player_positions,
            start_bonus_orders,
            rew_shaping_params,
            layout_name,
            start_all_orders,
            num_items_for_soup,
            order_bonus,
            start_state,
            **kwargs,
        )
        
        self.observation_ranges = []
        for player_id in range(self.num_players):
            self.observation_ranges.append({"xmin": 0,
                                            "xmax": self.width,
                                            "ymin": 0,
                                            "ymax": self.height})
            
        if observation_ranges is not None:
            for player_id in range(self.num_players):
                if observation_ranges[player_id]["xmin"] is not None:
                    self.observation_ranges[player_id]["xmin"] = observation_ranges[player_id]["xmin"]
                if observation_ranges[player_id]["xmax"] is not None:
                    self.observation_ranges[player_id]["xmax"] = observation_ranges[player_id]["xmax"]
                if observation_ranges[player_id]["ymin"] is not None:
                    self.observation_ranges[player_id]["ymin"] = observation_ranges[player_id]["ymin"]
                if observation_ranges[player_id]["ymax"] is not None:
                    self.observation_ranges[player_id]["ymax"] = observation_ranges[player_id]["ymax"]
    
    @staticmethod
    def from_layout_name(layout_name, **params_to_overwrite):
        """
        Generates a OvercookedGridworld instance from a layout file.

        One can overwrite the default mdp configuration using partial_mdp_config.
        """
        params_to_overwrite = params_to_overwrite.copy()
        
        layout_path = os.path.join("src/envs/overcooked/layout", layout_name + ".layout")
        with open(layout_path, "r") as f:
            base_layout_params = eval(f.read())

        grid = base_layout_params["grid"]
        del base_layout_params["grid"]
        base_layout_params["layout_name"] = layout_name
        if "start_state" in base_layout_params:
            base_layout_params["start_state"] = OvercookedState.from_dict(
                base_layout_params["start_state"]
            )

        # Clean grid
        grid = [layout_row.strip() for layout_row in grid.split("\n")]
        return OvercookedGridworld_.from_grid(
            grid, base_layout_params, params_to_overwrite
        )
    
    @staticmethod
    def from_grid(
        layout_grid, base_layout_params={}, params_to_overwrite={}, debug=False
    ):
        """
        Returns instance of OvercookedGridworld with terrain and starting
        positions derived from layout_grid.
        One can override default configuration parameters of the mdp in
        partial_mdp_config.
        """
        mdp_config = copy.deepcopy(base_layout_params)

        layout_grid = [[c for c in row] for row in layout_grid]
        OvercookedGridworld._assert_valid_grid(layout_grid)

        if "layout_name" not in mdp_config:
            layout_name = "|".join(["".join(line) for line in layout_grid])
            mdp_config["layout_name"] = layout_name

        player_positions = [None] * 9
        for y, row in enumerate(layout_grid):
            for x, c in enumerate(row):
                if c in ["1", "2", "3", "4", "5", "6", "7", "8", "9"]:
                    layout_grid[y][x] = " "

                    # -1 is to account for fact that player indexing starts from 1 rather than 0
                    assert (
                        player_positions[int(c) - 1] is None
                    ), "Duplicate player in grid"
                    player_positions[int(c) - 1] = (x, y)

        num_players = len([x for x in player_positions if x is not None])
        player_positions = player_positions[:num_players]

        # After removing player positions from grid we have a terrain mtx
        mdp_config["terrain"] = layout_grid
        mdp_config["start_player_positions"] = player_positions

        for k, v in params_to_overwrite.items():
            curr_val = mdp_config.get(k, None)
            if debug:
                print(
                    "Overwriting mdp layout standard config value {}:{} -> {}".format(
                        k, curr_val, v
                    )
                )
            mdp_config[k] = v

        return OvercookedGridworld_(**mdp_config)
    
    def resolve_interacts(self, new_state, joint_action, events_infos):
        """
        Resolve any INTERACT actions, if present.

        Currently if two players both interact with a terrain, we resolve player 1's interact
        first and then player 2's, without doing anything like collision checking.
        """
        pot_states = self.get_pot_states(new_state)
        # We divide reward by agent to keep track of who contributed
        sparse_reward, shaped_reward = (
            [0] * self.num_players,
            [0] * self.num_players,
        )

        for player_idx, (player, action) in enumerate(
            zip(new_state.players, joint_action)
        ):

            if action != Action.INTERACT:
                continue

            pos, o = player.position, player.orientation
            i_pos = Action.move_in_direction(pos, o)
            terrain_type = self.get_terrain_type_at_pos(i_pos)

            # NOTE: we always log pickup/drop before performing it, as that's
            # what the logic of determining whether the pickup/drop is useful assumes
            if terrain_type == "X":

                if player.has_object() and not new_state.has_object(i_pos):
                    obj_name = player.get_object().name
                    self.log_object_drop(
                        events_infos,
                        new_state,
                        obj_name,
                        pot_states,
                        player_idx,
                    )
                    shaped_reward[player_idx] += self.reward_shaping_params[
                        "OBJECT_DROP_REW"
                    ]

                    # Drop object on counter
                    obj = player.remove_object()
                    new_state.add_object(obj, i_pos)

                elif not player.has_object() and new_state.has_object(i_pos):
                    obj_name = new_state.get_object(i_pos).name
                    self.log_object_pickup(
                        events_infos,
                        new_state,
                        obj_name,
                        pot_states,
                        player_idx,
                    )
                    shaped_reward[player_idx] += self.reward_shaping_params[
                        "OBJECT_PICKUP_REW"
                    ]

                    # Pick up object from counter
                    obj = new_state.remove_object(i_pos)
                    player.set_object(obj)

            elif terrain_type == "O" and player.held_object is None:
                self.log_object_pickup(
                    events_infos, new_state, "onion", pot_states, player_idx
                )
                shaped_reward[player_idx] += self.reward_shaping_params[
                    "OBJECT_PICKUP_REW"
                ]

                # Onion pickup from dispenser
                obj = ObjectState("onion", pos)
                player.set_object(obj)

            elif terrain_type == "T" and player.held_object is None:
                self.log_object_pickup(
                    events_infos, new_state, "tomato", pot_states, player_idx
                )
                shaped_reward[player_idx] += self.reward_shaping_params[
                    "OBJECT_PICKUP_REW"
                ]
                # Tomato pickup from dispenser
                player.set_object(ObjectState("tomato", pos))

            elif terrain_type == "D" and player.held_object is None:
                self.log_object_pickup(
                    events_infos, new_state, "dish", pot_states, player_idx
                )
                shaped_reward[player_idx] += self.reward_shaping_params[
                    "OBJECT_PICKUP_REW"
                ]

                # Give shaped reward if pickup is useful
                if self.is_dish_pickup_useful(new_state, pot_states):
                    shaped_reward[player_idx] += self.reward_shaping_params[
                        "DISH_PICKUP_REWARD"
                    ]

                # Perform dish pickup from dispenser
                obj = ObjectState("dish", pos)
                player.set_object(obj)

            elif terrain_type == "P" and not player.has_object():
                # Cooking soup
                if self.soup_to_be_cooked_at_location(new_state, i_pos):
                    soup = new_state.get_object(i_pos)
                    soup.begin_cooking()
                    soup.auto_finish()  # soup finish immediately

            elif terrain_type == "P" and player.has_object():

                if (
                    player.get_object().name == "dish"
                    and self.soup_ready_at_location(new_state, i_pos)
                ):
                    self.log_object_pickup(
                        events_infos, new_state, "soup", pot_states, player_idx
                    )

                    # Pick up soup
                    player.remove_object()  # Remove the dish
                    obj = new_state.remove_object(i_pos)  # Get soup
                    player.set_object(obj)
                    shaped_reward[player_idx] += self.reward_shaping_params[
                        "SOUP_PICKUP_REWARD"
                    ]

                elif player.get_object().name in Recipe.ALL_INGREDIENTS:
                    # Adding ingredient to soup

                    if not new_state.has_object(i_pos):
                        # Pot was empty, add soup to it
                        new_state.add_object(SoupState(i_pos, ingredients=[]))

                    # Add ingredient if possible
                    soup = new_state.get_object(i_pos)
                    if not soup.is_full:
                        old_soup = soup.deepcopy()
                        obj = player.remove_object()
                        soup.add_ingredient(obj)
                        shaped_reward[
                            player_idx
                        ] += self.reward_shaping_params["PLACEMENT_IN_POT_REW"]

                        # Log potting
                        self.log_object_potting(
                            events_infos,
                            new_state,
                            old_soup,
                            soup,
                            obj.name,
                            player_idx,
                        )
                        if obj.name == Recipe.ONION:
                            events_infos["potting_onion"][player_idx] = True

            elif terrain_type == "S" and player.has_object():
                obj = player.get_object()
                if obj.name == "soup":

                    delivery_rew = self.deliver_soup(new_state, player, obj)
                    delivery_rew = 100 if delivery_rew else 0
                    sparse_reward[player_idx] += delivery_rew

                    # Log soup delivery
                    events_infos["soup_delivery"][player_idx] = True

        return sparse_reward, shaped_reward
    
    def featurize_state(self, overcooked_state, num_pots=1, **kwargs):
        """
        Encode state with some manually designed features. Works for arbitrary number of players

        Arguments:
            overcooked_state (OvercookedState): state we wish to featurize
            num_pots (int): Encode the state (ingredients, whether cooking or not, etc) of the 'num_pots' closest pots to each player.
                If i < num_pots pots are reachable by player i, then pots [i+1, num_pots] are encoded as all zeros. Changing this
                impacts the shape of the feature encoding

        Returns:
            ordered_features (list[np.Array]): The ith element contains a player-centric featurized view for the ith player

            The encoding for player i is as follows:

                [player_i_features, other_player_features player_i_dist_to_other_players, player_i_position]

                player_{i}_features (length num_pots*10 + 24):
                    pi_orientation: length 4 one-hot-encoding of direction currently facing
                    pi_obj: length 4 one-hot-encoding of object currently being held (all 0s if no object held)
                    pi_wall_{j}: {0, 1} boolean value of whether player i has wall immediately in direction j
                    pi_closest_{onion|tomato|dish|soup|serving|empty_counter}: (dx, dy) where dx = x dist to item, dy = y dist to item. (0, 0) if item is currently held
                    pi_cloest_soup_n_{onions|tomatoes}: int value for number of this ingredient in closest soup
                    pi_closest_pot_{j}_exists: {0, 1} depending on whether jth closest pot found. If 0, then all other pot features are 0. Note: can
                        be 0 even if there are more than j pots on layout, if the pot is not reachable by player i
                    pi_closest_pot_{j}_{is_empty|is_full|is_cooking|is_ready}: {0, 1} depending on boolean value for jth closest pot
                    pi_closest_pot_{j}_{num_onions|num_tomatoes}: int value for number of this ingredient in jth closest pot
                    pi_closest_pot_{j}_cook_time: int value for seconds remaining on soup. -1 if no soup is cooking
                    pi_closest_pot_{j}: (dx, dy) to jth closest pot from player i location

                other_player_features (length (num_players - 1)*(num_pots*10 + 24)):
                    ordered concatenation of player_{j}_features for j != i

                player_i_dist_to_other_players (length (num_players - 1)*2):
                    [player_j.pos - player_i.pos for j != i]

                player_i_position (length 2)
        """

        all_features = {}

        def concat_dicts(a, b):
            return {**a, **b}
        
        def make_closest_feature(idx, player, name, locations):
            """
            Compute (x, y) deltas to closest feature of type `name`, and save it in the features dict
            """
            feat_dict = {}
            obj = None
            held_obj = player.held_object
            held_obj_name = held_obj.name if held_obj else "none"
            if held_obj_name == name:
                obj = held_obj
                feat_dict["p{}_closest_{}".format(idx, name)] = (0, 0)
            else:
                loc, deltas = self.get_deltas_to_closest_location(player, locations)
                if loc and overcooked_state.has_object(loc):
                    obj = overcooked_state.get_object(loc)
                # rescale x, y to [-1, 1] of deltas
                x, y = deltas
                deltas = (x / (self.observation_ranges[idx]["xmax"] - self.observation_ranges[idx]["xmin"]),
                          y / (self.observation_ranges[idx]["ymax"] - self.observation_ranges[idx]["ymin"]))
                feat_dict["p{}_closest_{}".format(idx, name)] = deltas

            if name == "soup":
                num_onions = num_tomatoes = 0
                if obj:
                    ingredients_cnt = Counter(obj.ingredients)
                    num_onions, num_tomatoes = (
                        ingredients_cnt["onion"],
                        ingredients_cnt["tomato"],
                    )
                feat_dict["p{}_closest_soup_n_onions".format(idx)] = [num_onions]
                feat_dict["p{}_closest_soup_n_tomatoes".format(idx)] = [num_tomatoes]

            return feat_dict
        
        def make_pot_feature(idx, player, pot_idx, pot_loc, pot_states):
            """
            Encode pot at pot_loc relative to 'player'
            """
            # Pot doesn't exist
            feat_dict = {}
            if not pot_loc:
                feat_dict["p{}_closest_pot_{}_exists".format(idx, pot_idx)] = [
                    0
                ]
                feat_dict[
                    "p{}_closest_pot_{}_is_empty".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_is_full".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_is_cooking".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_is_ready".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_num_onions".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_num_tomatoes".format(idx, pot_idx)
                ] = [0]
                feat_dict[
                    "p{}_closest_pot_{}_cook_time".format(idx, pot_idx)
                ] = [0]
                feat_dict["p{}_closest_pot_{}".format(idx, pot_idx)] = (0, 0)
                return feat_dict

            # Get position information
            x, y = self.get_deltas_to_location(player, pot_loc)
            deltas = (x / (self.observation_ranges[idx]["xmax"] - self.observation_ranges[idx]["xmin"]),
                      y / (self.observation_ranges[idx]["ymax"] - self.observation_ranges[idx]["ymin"]))

            # Get pot state info
            is_empty = int(pot_loc in self.get_empty_pots(pot_states))
            is_full = int(pot_loc in self.get_full_pots(pot_states))
            is_cooking = int(pot_loc in self.get_cooking_pots(pot_states))
            is_ready = int(pot_loc in self.get_ready_pots(pot_states))

            # Get soup state info
            num_onions = num_tomatoes = 0
            cook_time_remaining = 0
            if not is_empty:
                soup = overcooked_state.get_object(pot_loc)
                ingredients_cnt = Counter(soup.ingredients)
                num_onions, num_tomatoes = (
                    ingredients_cnt["onion"],
                    ingredients_cnt["tomato"],
                )
                cook_time_remaining = (
                    0 if soup.is_idle else soup.cook_time_remaining
                )

            # Encode pot and soup info
            feat_dict["p{}_closest_pot_{}_exists".format(idx, pot_idx)] = [1]
            feat_dict["p{}_closest_pot_{}_is_empty".format(idx, pot_idx)] = [
                is_empty
            ]
            feat_dict["p{}_closest_pot_{}_is_full".format(idx, pot_idx)] = [
                is_full
            ]
            feat_dict["p{}_closest_pot_{}_is_cooking".format(idx, pot_idx)] = [
                is_cooking
            ]
            feat_dict["p{}_closest_pot_{}_is_ready".format(idx, pot_idx)] = [
                is_ready
            ]
            feat_dict["p{}_closest_pot_{}_num_onions".format(idx, pot_idx)] = [
                num_onions
            ]
            feat_dict[
                "p{}_closest_pot_{}_num_tomatoes".format(idx, pot_idx)
            ] = [num_tomatoes]
            feat_dict["p{}_closest_pot_{}_cook_time".format(idx, pot_idx)] = [
                cook_time_remaining
            ]
            feat_dict["p{}_closest_pot_{}".format(idx, pot_idx)] = deltas

            return feat_dict      

        IDX_TO_OBJ = ["onion", "soup", "dish", "tomato"]
        OBJ_TO_IDX = {o_name: idx for idx, o_name in enumerate(IDX_TO_OBJ)}

        counter_objects = self.get_counter_objects_dict(overcooked_state)
        pot_states = self.get_pot_states(overcooked_state)
        
        for i, player in enumerate(overcooked_state.players):
            # Player info
            orientation_idx = Direction.DIRECTION_TO_INDEX[player.orientation]
            all_features["p{}_orientation".format(i)] = np.eye(4)[
                orientation_idx
            ]
            obj = player.held_object

            if obj is None:
                held_obj_name = "none"
                all_features["p{}_objs".format(i)] = np.zeros(len(IDX_TO_OBJ))
            else:
                held_obj_name = obj.name
                obj_idx = OBJ_TO_IDX[held_obj_name]
                all_features["p{}_objs".format(i)] = np.eye(len(IDX_TO_OBJ))[
                    obj_idx
                ]
                
            # Closest feature for each object type
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i,
                    player,
                    "onion",
                    self.get_objects_in_observation_range(
                        self.get_onion_dispenser_locations()
                        + counter_objects["onion"], i)
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i,
                    player,
                    "tomato",
                    self.get_objects_in_observation_range(
                        self.get_tomato_dispenser_locations()
                        + counter_objects["tomato"], i)
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i,
                    player,
                    "dish",
                    self.get_objects_in_observation_range(
                        self.get_dish_dispenser_locations()
                        + counter_objects["dish"], i)
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i, player, "soup", 
                    self.get_objects_in_observation_range(
                        counter_objects["soup"], i)
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i, player, "serving", 
                    self.get_objects_in_observation_range(
                        self.get_serving_locations(), i)
                ),
            )
            all_features = concat_dicts(
                all_features,
                make_closest_feature(
                    i,
                    player,
                    "empty_counter",
                    self.get_objects_in_observation_range(
                        self.get_empty_counter_locations(overcooked_state), i)
                ),
            )
            
            # Closest pots info
            pot_locations = self.get_objects_in_observation_range(self.get_pot_locations().copy(), i)
            for pot_idx in range(num_pots):
                closest_pot_loc, _ = self.get_deltas_to_closest_location(player, pot_locations)
                pot_features = make_pot_feature(
                    i, player, pot_idx, closest_pot_loc, pot_states
                )
                all_features = concat_dicts(all_features, pot_features)

                if closest_pot_loc:
                    pot_locations.remove(closest_pot_loc)

            # Adjacent features info
            for direction, pos_and_feat in enumerate(
                self.get_adjacent_features(player)
            ):
                _, feat = pos_and_feat
                all_features["p{}_wall_{}".format(i, direction)] = (
                    [0] if feat == " " else [1]
                )
                
        # Convert all list and tuple values to np.arrays
        features_np = {k: np.array(v) for k, v in all_features.items()}
        
        final_features = []
        # Compute all player-centric features for each player
        for i, player_i in enumerate(overcooked_state.players):
            # All absolute player-centric features
            player_i_dict = {
                k: v
                for k, v in features_np.items()
                if k[:2] == "p{}".format(i)
            }
            features = np.concatenate(list(player_i_dict.values()), dtype=np.float32)
            
            abs_pos = np.array(player_i.position, dtype=np.float32)
            abs_pos[0] = abs_pos[0] / self.width
            abs_pos[1] = abs_pos[1] / self.height
            
            final_features.append(np.concatenate([abs_pos, features]))

        return final_features

    def get_deltas_to_closest_location(self, player, locations):
        if len(locations) == 0:
            return None, (0, 0)
        
        deltas = [self.get_deltas_to_location(player, loc) for loc in locations]
        dists = [abs(dx) + abs(dy) for dx, dy in deltas]
        index = np.argmin(dists)
        
        return locations[index], deltas[index]
    
    def get_objects_in_observation_range(self, objects_list, player_id):
        objects_in_range = []
        for x, y in objects_list:
            if self.is_in_observation_range(x, y, player_id):
                objects_in_range.append((x, y))
        return objects_in_range
                
    def is_in_observation_range(self, x, y, player_id):
        return (self.observation_ranges[player_id]["xmin"] <= x <=
                self.observation_ranges[player_id]["xmax"]) and \
               (self.observation_ranges[player_id]["ymin"] <= y <=
                self.observation_ranges[player_id]["ymax"])