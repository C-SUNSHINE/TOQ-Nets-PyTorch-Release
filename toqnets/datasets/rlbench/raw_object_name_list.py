#! /usr/bin/env python3
# -*- coding: utf-8 -*-
# File   : raw_object_name_list.py
# Author : Zhezheng Luo
# Email  : luozhezheng@gmail.com
# Date   : 08/02/2021
#
# This file is part of TOQ-Nets-PyTorch.
# Distributed under terms of the MIT license.

object_name_list = ['Cuboid', 'Force_sensor', 'Plane', 'Shape1', 'Shape_sub', 'Shape_sub0', 'Shape_sub1', 'base',
                    'base_visual', 'bin', 'bin_visual', 'bottle', 'bottle_detector', 'bottle_visual', 'bottom_joint',
                    'boundary', 'boundary_root', 'box_base', 'cabinet_base', 'cap', 'cap_detector', 'cap_visual',
                    'chocolate_jello', 'chocolate_jello_grasp_point', 'chocolate_jello_visual', 'coffee',
                    'coffee_grasp_point', 'coffee_visual', 'crackers', 'crackers_grasp_point', 'crackers_visual', 'cup',
                    'cup1', 'cup1_visual', 'cup2', 'cup2_visual', 'cup_visual', 'cupboard', 'dish_rack',
                    'dish_rack_pillar0', 'dish_rack_pillar1', 'dish_rack_pillar2', 'dish_rack_pillar3',
                    'dish_rack_pillar4', 'dish_rack_pillar5', 'door_bottom', 'door_bottom_visual', 'door_top',
                    'door_top_visual', 'drawer_bottom', 'drawer_frame', 'drawer_joint_bottom', 'drawer_joint_middle',
                    'drawer_joint_top', 'drawer_legs', 'drawer_middle', 'drawer_top', 'dummy', 'fork', 'fork_detector',
                    'fork_visual', 'fridge_base', 'fridge_base_visual', 'fridge_root', 'fridge_visual', 'glass',
                    'glass_detector', 'glass_visual', 'grill', 'grill_root', 'grill_visual', 'handle_visual', 'holder',
                    'holder_visual', 'item', 'jar0', 'jar1', 'jar_lid0', 'joint', 'knife', 'knife_detector',
                    'knife_visual', 'label', 'laptop_holder', 'left_board', 'left_board_visual', 'left_close_waypoint',
                    'left_far_waypoint', 'left_handle_visual', 'left_initial_waypoint', 'left_joint', 'lid',
                    'lid_visual', 'microwave', 'microwave_door', 'microwave_door_joint', 'microwave_door_resp',
                    'microwave_frame_resp', 'microwave_frame_vis', 'mustard', 'mustard_grasp_point', 'mustard_visual',
                    'oven', 'oven_base', 'oven_boundary_root', 'oven_door', 'oven_door_joint', 'oven_door_visual',
                    'oven_frame0', 'oven_gas_source0', 'oven_knob', 'oven_knob_10', 'oven_knob_11', 'oven_knob_12',
                    'oven_knob_13', 'oven_knob_14', 'oven_knob_8', 'oven_knob_joint', 'oven_knob_visual',
                    'oven_stove_top0', 'oven_tray', 'oven_tray_boundary', 'oven_tray_joint', 'oven_tray_visual',
                    'place_cup_left_waypoint', 'plate', 'plate_detector', 'plate_target', 'plate_visual',
                    'push_buttons_boundary', 'push_buttons_target0', 'push_buttons_target1', 'push_buttons_target2',
                    'right_board', 'right_board_visual', 'right_handle_visual', 'right_joint', 'rubbish',
                    'rubbish_visual', 'scoop_with_spatula_spatula', 'sensor_grill_top', 'sensor_handle', 'soup',
                    'soup_grasp_point', 'soup_visual', 'spam', 'spam_grasp_point', 'spam_visual', 'spatula_visual',
                    'spawn_boundary', 'spoon', 'spoon_detector', 'spoon_visual', 'stand_base', 'strawberry_jello',
                    'strawberry_jello_grasp_point', 'strawberry_jello_visual', 'success', 'success_bottom',
                    'success_middle', 'success_pos0', 'success_pos1', 'success_pos2', 'success_source',
                    'success_target', 'success_top', 'sugar', 'sugar_grasp_point', 'sugar_visual', 'surface', 'switch',
                    'switch_main', 'switch_visual', 'target_button_joint0', 'target_button_joint1',
                    'target_button_joint2', 'target_button_topPlate0', 'target_button_topPlate1',
                    'target_button_topPlate2', 'target_button_wrap0', 'target_button_wrap1', 'target_button_wrap2',
                    'task_wall', 'toilet_roll', 'toilet_roll_holder', 'toilet_roll_visual', 'tomato1', 'tomato1_visual',
                    'tomato2', 'tomato2_visual', 'tongue', 'tongue_visual', 'top_joint', 'tray', 'tray_visual', 'tuna',
                    'tuna_grasp_point', 'tuna_visual', 'waypoint0', 'waypoint1', 'waypoint10', 'waypoint11',
                    'waypoint12', 'waypoint13', 'waypoint14', 'waypoint15', 'waypoint16', 'waypoint17', 'waypoint18',
                    'waypoint19', 'waypoint2', 'waypoint20', 'waypoint21', 'waypoint22', 'waypoint23', 'waypoint24',
                    'waypoint25', 'waypoint26', 'waypoint27', 'waypoint3', 'waypoint4', 'waypoint5', 'waypoint6',
                    'waypoint7', 'waypoint8', 'waypoint9', 'waypoint_anchor_bottom', 'waypoint_anchor_middle',
                    'waypoint_anchor_top']

def get_raw_object_name_list():
    return object_name_list[:]