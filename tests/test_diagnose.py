#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.


import unittest
import numpy as np

from deepxtrace import diagnose as ds


class TestDiagnose(unittest.TestCase):
    def setUp(self):
        self.abnormal_row = np.array([
            [934, 1501, 764, 463, 526, 526, 1553, 1102],
            [488, 501, 517, 485, 497, 547, 484, 517],
            [934, 1501, 764, 463, 526, 526, 1553, 1102],
            [19779445, 19779792, 19773172, 493, 19779409, 19779527, 535, 19779431],
            [488, 501, 517, 485, 497, 547, 484, 517],
            [684, 440, 230, 476, 197, 478, 538, 531],
            [488, 501, 517, 485, 497, 547, 484, 517],
            [488, 501, 517, 485, 497, 547, 484, 517],

        ])
        self.abnormal_col = np.array([
            [449, 637, 19776694, 480, 19779137, 487, 466, 411],
            [684, 440, 19751230, 476, 19779054, 478, 538, 531],
            [488, 501, 517, 485, 19779234, 547, 484, 517],
            [1263, 2052, 19757762, 440, 19779484, 476, 2602, 2947],
            [934, 1501, 19776764, 463, 526, 526, 1553, 1102],
            [2343, 1807, 19779054, 486, 19779895, 517, 1979, 2407],
            [445, 792, 19773172, 493, 19779409, 527, 535, 431],
            [477, 663, 19774488, 506, 19779483, 604, 555, 474],
        ])

        self.abnormal_mixed = np.array([
            [449, 637, 19776694, 480, 19779137, 486, 466, 411],
            [684, 440, 19751230, 476, 19779054, 478, 538, 531],
            [488, 501, 517, 485, 19779234, 947, 484, 517],
            [1263, 2052, 19757762, 440, 19779484, 476, 2602, 2947],
            [934, 1501, 19776764, 463, 526, 526, 1553, 1102],
            [2343, 1807, 19779054, 486, 19779895, 517, 1979, 2407],
            [19779445, 19779792, 19773172, 19779493,
                19779409, 19779527, 535, 19779431],
            [477, 663, 19774488, 506, 19779483, 604, 555, 474],
        ])

        self.abnormal_point = np.array([
            [16, 13, 11, 17, 18, 18, 19, 12],
            [10, 19, 11, 18, 16, 16, 16, 13],
            [18, 18, 11, 10, 18, 19, 18, 13],
            [13, 18, 11, 11, 125, 11, 18, 18],
            [14, 20, 11, 14, 14, 16, 18, 16],
            [20, 20, 11, 14, 19, 13, 15, 18],
            [38, 17, 13, 16, 13, 13, 13, 13],
            [15, 17, 12, 18, 13, 13, 15, 14],
        ])

        self.mc2_layered = np.array([
            [169, 537, 530, 294, 173, 128, 139, 140,
                40, 0, 0, 0, 0, 0, 0, 0],
            [1617, 196, 207, 170, 187, 151, 887, 174,
                0, 34, 0, 0, 0, 0, 0, 0],
            [1626, 210, 194, 186, 174, 162, 864, 160,
                0, 0, 31, 0, 0, 0, 0, 0],
            [1635, 324, 341, 186, 178, 153, 866, 169,
                0, 0, 0, 34, 0, 0, 0, 0],
            [1635, 543, 534, 302, 176, 125, 847, 140,
                0, 0, 0, 0, 33, 0, 0, 0],
            [1712, 681, 671, 401, 232, 102, 877, 132,
                0, 0, 0, 0, 0, 37, 0, 0],
            [997, 656, 643, 382, 235, 172, 107, 146, 0,
                0, 0, 0, 0, 0, 42, 0],
            [1918, 941, 931, 652, 448, 314, 1064, 199,
                0, 0, 0, 0, 0, 0, 0, 42],
            [1480, 0, 0, 0, 0, 0, 0, 0, 167, 239, 343,
                154, 148, 150, 155, 143],
            [0, 46, 0, 0, 0, 0, 0, 0, 1599, 169, 237,
                156, 149, 146, 860, 140],
            [0, 0, 48, 0, 0, 0, 0, 0, 1610, 161, 168,
                159, 150, 161, 846, 145],
            [0, 0, 0, 41, 0, 0, 0, 0, 1687, 320, 452,
                82, 139, 166, 875, 136],
            [0, 0, 0, 0, 42, 0, 0, 0, 1802, 481, 616,
                242, 168, 214, 918, 166],
            [0, 0, 0, 0, 0, 35, 0, 0, 1746, 417, 559,
                226, 171, 185, 903, 151],
            [0, 0, 0, 0, 0, 0, 738, 0, 1011, 393, 529,
                171, 150, 162, 176, 154],
            [0, 0, 0, 0, 0, 0, 0, 36, 1866, 555, 693,
                325, 211, 222, 965, 180]
        ])

    def test_diagnose_row(self):
        res = ds.Diagnose.diagnose_matrix(self.abnormal_row)
        self.assertEqual(
            res, {
                'abnormal_cols': [], 'abnormal_rows': [
                    [3, 14833975.5, 7.997677898368585]], 'abnormal_points': []})

    def test_diagnose_col(self):
        res = ds.Diagnose.diagnose_matrix(self.abnormal_col)
        self.assertEqual(res,
                         {'abnormal_cols': [[2,
                                             17298710.125,
                                             3.9984464415004988],
                                            [4,
                                             17307027.75,
                                             4.000368988201534]],
                             'abnormal_rows': [],
                             'abnormal_points': []})

    def test_diagnose_mixed(self):
        res = ds.Diagnose.diagnose_matrix(
            self.abnormal_mixed, thres_col=1.0, thres_row=1.0)
        self.assertEqual(
            res, {
                'abnormal_cols': [
                    [2, 17298710.125, 2.946167089439367], [4, 17307027.75, 2.9475836755813525]], 'abnormal_rows': [
                    [6, 17306350.5, 2.9474683322033255]], 'abnormal_points': []})

    def test_diagnose_point(self):
        res = ds.Diagnose.diagnose_matrix(self.abnormal_point)
        self.assertEqual(
            res, {
                'abnormal_cols': [], 'abnormal_rows': [], 'abnormal_points': [
                    [3, 4, 125, 7.279344854723584]]})

    def test_mc2_layered(self):
        res = ds.Diagnose.diagnose_matrix(
            mat=self.mc2_layered, excluding_zeros=0)
        self.assertEqual(
            res, {
                'abnormal_cols': [
                    [
                        0, 799.3125, 3.2102414457222475]], 'abnormal_rows': [], 'abnormal_points': [
                    [
                        9, 8, 1599, 6.421988986422549], [
                            10, 8, 1610, 6.466167772445468], [
                                11, 8, 1687, 6.775419274605904], [
                                    12, 8, 1802, 7.237288401209152], [
                                        13, 8, 1746, 7.012378217819744], [
                                            15, 8, 1866, 7.494328610797046]]})

        res = ds.Diagnose.diagnose_matrix(
            mat=self.mc2_layered, excluding_zeros=1)
        self.assertEqual(res,
                         {'abnormal_cols': [[0,
                                             799.3125,
                                             3.2102414457222475]],
                          'abnormal_rows': [],
                             'abnormal_points': []})


if __name__ == '__main__':
    unittest.main()
