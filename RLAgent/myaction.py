from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

import numpy as np
from attr import define
from gymnasium import Space, spaces
from negmas.gb.common import ResponseType, field
from negmas.helpers import distribute_integer_randomly
from negmas.outcomes.issue_ops import itertools
from negmas.sao.common import SAOResponse

from scml.oneshot.awi import OneShotAWI
from scml.oneshot.context import BaseContext
from scml.scml2019.common import QUANTITY
from scml.oneshot.rl.helpers import recover_offers, encode_given_offers

from scml.oneshot.rl.action import FlexibleActionManager, ActionManager

@define(frozen=True)
class MyActionManager(ActionManager):
    """
    An action manager that matches any context.

    Args:
        n_prices: Number of distinct prices allowed in the action.
        max_quantity: Maximum allowed quantity to offer in any negotiation. The number of quantities is one plus that because zero is allowed to model ending negotiation.
        n_partners: Maximum of partners allowed in the action.

    Remarks:
        - This action manager will always generate offers that are within the
          price and quantity limits given in its parameters. Wen decoding them,
          it will scale them up so that the maximum corresponds to the actual value
          in the world it finds itself. For example, if `n_prices` is 10 and the world
          has only two prices currently in the price issue, it will use any value less than
          5 as the minimum price and any value above 5 as the maximum price. If on the other
          hand the current price issue has 20 values, then it will scale by multiplying the
          number given in the encoded action (ranging from 0 to 9) by 19/9 which makes it range
          from 0 to 19 which is what is expected by the world.
        - This action manager will adjust offers for different number of partners as follows:
          - If the true number of partners is larger than `n_partners` used by this action manager,
            it will simply use `n_partners` of them and always end negotiations with the rest of them.
          - If the true number of partners is smaller than `n_partners`, it will use the first `n_partners`
            values in the encoded action and increase the quantities of any counter offers (i.e. ones in
            which the response is REJECT_OFFER) by the amount missing from the ignored partners in the encoded
            action up to the maximum quantities allowed by the current negotiation context. For example, if
            `n_partneers` is 4 and we have only 2 partners in reality, and the received quantities from
            partners were [4, 3] while the maximum quantity allowed is 10 and the encoded action was
            [2, *, 3, *, 2, *, 1, *] (where we ignored prices), then the encoded action will be converted to
            [(Reject, 5, *), (Accept, 3, *)] where the 3 extra units that were supposed to be offered to the
            last two partners are moved to the first partner. If the maximum quantity allowed was 4 in that
            example, the result will be [(Reject, 4, *), (Accept, 3, *)].

    """

    capacity_multiplier: int = 1
    n_prices: int = 2 # 価格は2値であるから？
    max_group_size: int = 2
    reduce_space_size: bool = True
    extra_checks: bool = False
    max_quantity: int = field(init=False, default=10)

    mode: int = 0  # 0: train専用, 1: try_a_model専用
    # rule_based_agent: CustomCautiousOneShotAgent = field(init=False, default=None)
    rule_based_agent: CustomEqualDistOneShotAgent = field(init=False, default=None)

    def __attrs_post_init__(self):
        p = self.context.extract_context_params(self.reduce_space_size)
        if p.nlines:
            object.__setattr__(
                self, "max_quantity", self.capacity_multiplier * p.nlines
            )
            object.__setattr__(self, "n_consumers", p.nconsumers)
            object.__setattr__(self, "n_suppliers", p.nsuppliers)
        object.__setattr__(self, "n_partners", self.n_suppliers + self.n_consumers)

    def make_space(self) -> spaces.MultiDiscrete | spaces.Box:
        """Creates the action space"""
        return (
            spaces.MultiDiscrete(
                # np.asarray(
                #     [self.max_quantity + 1, self.n_prices] * self.n_partners
                # ).flatten()
                np.asarray(
                    [3, self.n_prices] * self.n_partners # 0, 1, 2の数量と0, 1の価格
                ).flatten()
            )
            if not self.continuous
            else spaces.Box(0.0, 1.0, shape=(self.n_partners * 2,))
        )

    def decode(self, awi: OneShotAWI, action: np.ndarray) -> dict[str, SAOResponse]:
        """
        Generates offers to all partners from an encoded action. Default is to return the action as it is assuming it is a `dict[str, SAOResponse]`
        """
        if self.mode == 1 and self.rule_based_agent:
            # try_a_model専用ロジック: ルールベースエージェントを使用
            self.rule_based_agent.init()
            offers = awi.current_offers
            states = awi.current_states
            response = self.rule_based_agent.counter_all(offers, states)
            if awi.level == 0:
                result = self.process_offers(
                    list(awi.my_consumers),
                    response
                )
            else:
                result = self.process_offers(
                    list(awi.my_suppliers),
                    response
                )
            action = result + action



        action = action.reshape((action.size // 2, 2)) 
        action[:, 0] = np.maximum(action[:, 0] - 1, 0) # convert to -1, 0, 1 for quantity and 0, 1 for price
        # print(f"Action: {action}")

        if not (len(action) == self.n_partners):
            raise AssertionError(
                f"{len(action)=}, {self.n_partners=} ({self.n_suppliers=}, {self.n_consumers=})"
            )
        offers = recover_offers(
            action,
            awi,
            self.n_suppliers,
            self.n_consumers,
            self.max_group_size,
            self.continuous,
            self.n_prices,
        )
        # print(f"decode_Action_offers: {offers}")
        separated_offers, responses = dict(), dict()
        nmis = awi.current_nmis
        for k, v in offers.items():
            if "+" not in k:
                separated_offers[k] = tuple(int(_) for _ in v) if v else v
                continue
            partners = k.split("+")
            if v is None:
                separated_offers |= dict(zip(partners, itertools.repeat(None)))
                continue
            q = v[QUANTITY]
            dist = distribute_integer_randomly(q, len(partners), 1)
            separated_offers |= dict(zip(partners, ((_, v[1], v[-1]) for _ in dist)))

        for k, v in separated_offers.items():
            nmi = nmis.get(k, None)
            if nmi is None:
                continue
            if v is None:
                responses[k] = SAOResponse(ResponseType.END_NEGOTIATION, None)
                continue
            partner_offer = nmi.state.current_offer  # type: ignore
            if v == partner_offer:
                responses[k] = SAOResponse(ResponseType.ACCEPT_OFFER, partner_offer)
                continue
            responses[k] = SAOResponse(ResponseType.REJECT_OFFER, v)

        # print(f"decode_Action_responses: {responses}")
        return responses

    def encode(self, awi: OneShotAWI, responses: dict[str, SAOResponse]) -> np.ndarray:
        """
        Receives offers for all partners and generates the corresponding action. Used mostly for debugging and testing.
        """
        offers = dict()
        for k, v in responses.items():
            if v.response == ResponseType.END_NEGOTIATION:
                offers[k] = None
                continue
            offers[k] = v.outcome
        encoded = encode_given_offers(
            offers,
            awi,
            self.n_suppliers,
            self.n_consumers,
            self.max_group_size,
            self.continuous,
        )

        return np.asarray(encoded, dtype=np.float32 if self.continuous else np.int32)

    def process_offers(self, partners, responses):
        """
        Processes offers into groups and calculates the final array.

        Args:
            partners (list[str]): List of partner IDs.
            responses (dict[str, SAOResponse]): Dictionary of responses for each partner.

        Returns:
            np.ndarray: Array in the format [quantity, price, quantity, price, ...].
        """
        # グループ数を定義
        num_groups = 4

        # グループを初期化
        groups = [[] for _ in range(num_groups)]

        # パートナーをグループに分割
        for i, partner in enumerate(partners):
            groups[i % num_groups].append(partner)

        # 各グループの数量を足し合わせ、価格をランダムに決定
        result = []
        for group in groups:
            total_quantity = sum(
                responses[partner].outcome[0]
                for partner in group
                if partner in responses and responses[partner].outcome is not None
            )
            random_price = random.randint(0, 0)  # 価格をランダムに決定 (例: 10〜20の範囲)
            result.extend([total_quantity, random_price])

        return np.array(result)

DefaultActionManager = MyActionManager
"""The default action manager"""

from gymnasium.core import ActionWrapper
from scml.oneshot.rl.env import OneShotEnv
from scml_agents.scml2024.oneshot.team_miyajima_oneshot.cautious import CautiousOneShotAgent, powerset#, distribute
from scml_agents.scml2024.oneshot.teamyuzuru.epsilon_greedy_agent import EpsilonGreedyAgent
from scml.oneshot.agents.rand import (
    EqualDistOneShotAgent,
    distribute
)
import random
TIME = 1
UNIT_PRICE = 2
class CustomCautiousOneShotAgent(CautiousOneShotAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def awi(self):
        return self._awi

    @awi.setter
    def awi(self, value):
        self._awi = value

    def counter_all(self, offers, states):
        response = dict()
        future_partners = {
            k for k, v in offers.items() if v[TIME] != self.awi.current_step
        }
        offers = {k: v for k, v in offers.items() if v[TIME] == self.awi.current_step}
        # process for sales and supplies independently
        for needs, all_partners, issues in [
            (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
            ),
            (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
            ),
        ]:
            # get a random price
            price = issues[UNIT_PRICE].rand()
            # find active partners in some random order
            partners = [_ for _ in all_partners if _ in offers.keys()]
            random.shuffle(partners)
            partners = set(partners)
            is_selling = all_partners == self.awi.my_consumers

            # ラウンドごとの相手一人あたりの平均提案個数を擬似的に算出->これ使ってないからいらない
            # if len(partners) > 0:
            #     neg_step = min(state.step for state in states.values())
            #     self.rounds_ave_offered[neg_step] = 0.7 * self.rounds_ave_offered[
            #         neg_step
            #     ] + 0.3 * sum([offers[p][QUANTITY] for p in partners]) / len(partners)

                # print("round",neg_step,self.rounds_ave_offered,sum([offers[p][QUANTITY] for p in partners]))

            unneeded_response = (
                SAOResponse(ResponseType.END_NEGOTIATION, None)
                if not self.awi.allow_zero_quantity
                else SAOResponse(
                    ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0)
                )
            )

            # find the set of partners that gave me the best offer set
            # (i.e. total quantity nearest to my needs)
            plist = list(powerset(partners))[::-1]
            plus_best_diff, plus_best_expected_diff, plus_best_indx = (
                float("inf"),
                float("inf"),
                -1,
            )
            minus_best_diff, minus_best_expected_diff, minus_best_indx = (
                -float("inf"),
                -float("inf"),
                -1,
            )
            best_diff, best_indx = float("inf"), -1

            for i, partner_ids in enumerate(plist):
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = offered - needs
                if diff >= 0:  # 必要以上の量のとき
                    if diff < plus_best_diff:
                        plus_best_diff, plus_best_indx = diff, i
                    elif diff == plus_best_diff:
                        if is_selling:  # 売り手の場合は高かったら更新
                            if sum(offers[p][UNIT_PRICE] for p in partner_ids) > sum(
                                offers[p][UNIT_PRICE] for p in plist[plus_best_indx]
                            ):
                                plus_best_diff, plus_best_indx = diff, i
                        else:  # 買い手の場合は安かったら更新
                            if sum(offers[p][UNIT_PRICE] for p in partner_ids) < sum(
                                offers[p][UNIT_PRICE] for p in plist[plus_best_indx]
                            ):
                                plus_best_diff, plus_best_indx = diff, i
                if diff <= 0:  # 必要量に満たないとき
                    if diff > minus_best_diff:
                        minus_best_diff, minus_best_indx = diff, i
                    elif diff == minus_best_diff:
                        if (
                            diff < 0 and len(partner_ids) < len(plist[minus_best_indx])
                        ):  # アクセプトする不足分をCounterOfferできる相手の数が多かったら更新
                            minus_best_diff, minus_best_indx = diff, i
                        elif diff == 0 or len(partner_ids) == len(
                            plist[minus_best_indx]
                        ):
                            if is_selling:  # 売り手の場合は高かったら更新
                                if sum(
                                    offers[p][UNIT_PRICE] for p in partner_ids
                                ) > sum(
                                    offers[p][UNIT_PRICE]
                                    for p in plist[minus_best_indx]
                                ):
                                    minus_best_diff, minus_best_indx = diff, i
                            else:  # 買い手の場合は安かったら更新
                                if sum(
                                    offers[p][UNIT_PRICE] for p in partner_ids
                                ) < sum(
                                    offers[p][UNIT_PRICE]
                                    for p in plist[minus_best_indx]
                                ):
                                    minus_best_diff, minus_best_indx = diff, i

            th_min_plus, th_max_plus = self._allowed_mismatch(
                min(state.relative_time for state in states.values()),
                len(partners.difference(plist[plus_best_indx]).union(future_partners)),
                is_selling,
            )
            th_min_minus, th_max_minus = self._allowed_mismatch(
                min(state.relative_time for state in states.values()),
                len(partners.difference(plist[minus_best_indx]).union(future_partners)),
                is_selling,
            )
            if th_min_minus <= minus_best_diff or plus_best_diff <= th_max_plus:
                if th_min_minus <= minus_best_diff and plus_best_diff <= th_max_plus:
                    if -minus_best_diff == plus_best_diff:
                        if is_selling:  # 売り手のときは、best_diff>0だとshortfall penaltyが発生するのでminus優先
                            best_diff, best_indx = minus_best_diff, minus_best_indx
                        else:  # 買い手のときは、best_diff<0だとshortfall penaltyが発生するのでplus優先
                            best_diff, best_indx = plus_best_diff, plus_best_indx
                    elif -minus_best_diff < plus_best_diff:
                        # 自身が買い手で、かつ不足分を残りの相手へのCounterOfferで補えないときは、shortfall penaltyを防ぐためplus優先
                        if (
                            not is_selling
                            and len(
                                partners.difference(plist[minus_best_indx]).union(
                                    future_partners
                                )
                            )
                            == 0
                        ):
                            best_diff, best_indx = plus_best_diff, plus_best_indx
                        else:
                            best_diff, best_indx = minus_best_diff, minus_best_indx
                    else:
                        best_diff, best_indx = plus_best_diff, plus_best_indx
                elif minus_best_diff < th_min_minus and plus_best_diff <= th_max_plus:
                    best_diff, best_indx = plus_best_diff, plus_best_indx
                else:
                    best_diff, best_indx = minus_best_diff, minus_best_indx

                partner_ids = plist[best_indx]
                others = list(partners.difference(partner_ids).union(future_partners))
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: unneeded_response for k in others}

                if (
                    best_diff < 0 and len(others) > 0
                ):  # 必要量に足りないとき、CounterOfferで補う
                    s, p = self._step_and_price(best_price=True)
                    t = min(states[p].relative_time for p in others)
                    offering_quanitity = (
                        int(-best_diff * (1 + self._overordering_fraction(t)))
                        if len(others) > 1
                        else -best_diff
                    )
                    if self.awi.current_step > self.awi.n_steps * 0.5:
                        concentrated_ids = sorted(
                            others,
                            key=lambda x: self.total_agreed_quantity[x],
                            reverse=True,
                        )[:1]
                        concentrated_idx = [
                            i for i, p in enumerate(others) if p in concentrated_ids
                        ]
                        distribution = dict(
                            zip(
                                others,
                                distribute(
                                    offering_quanitity,
                                    len(others),
                                    mx=self.awi.n_lines,
                                    concentrated=True,
                                    concentrated_idx=concentrated_idx,
                                ),
                            )
                        )
                    else:
                        distribution = dict(
                            zip(
                                others,
                                distribute(
                                    offering_quanitity, len(others), mx=self.awi.n_lines
                                ),
                            )
                        )
                    response.update(
                        {
                            k: (
                                unneeded_response
                                if q == 0
                                else SAOResponse(ResponseType.REJECT_OFFER, (q, s, p))
                            )
                            for k, q in distribution.items()
                        }
                    )

                continue

            # If I still do not have a good enough offer, distribute my current needs
            # randomly over my partners.
            t = min(_.relative_time for _ in states.values())
            # distribution = self.distribute_needs(t)

            partners = partners.union(future_partners)
            partners = list(partners)
            offering_quanitity = (
                int(needs * (1 + self._overordering_fraction(t)))
                if len(partners) > 1
                else needs
            )
            if self.awi.current_step > self.awi.n_steps * 0.5 and len(partners) > 0:
                concentrated_ids = sorted(
                    partners, key=lambda x: self.total_agreed_quantity[x], reverse=True
                )[:1]
                concentrated_idx = [
                    i for i, p in enumerate(partners) if p in concentrated_ids
                ]
                distribution = dict(
                    zip(
                        partners,
                        distribute(
                            offering_quanitity,
                            len(partners),
                            mx=self.awi.n_lines,
                            concentrated=True,
                            concentrated_idx=concentrated_idx,
                        ),
                    )
                )
            else:
                distribution = dict(
                    zip(
                        partners,
                        distribute(
                            offering_quanitity, len(partners), mx=self.awi.n_lines
                        ),
                    )
                )

            response.update(
                {
                    k: (
                        unneeded_response
                        if q == 0
                        else SAOResponse(
                            ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
                        )
                    )
                    for k, q in distribution.items()
                }
            )
        return response

class CustomEpsilonGreedyAgent(EpsilonGreedyAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def awi(self):
        return self._awi

    @awi.setter
    def awi(self, value):
        self._awi = value

class CustomEqualDistOneShotAgent(EqualDistOneShotAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    @property
    def awi(self):
        return self._awi

    @awi.setter
    def awi(self, value):
        self._awi = value

    def distribute_needs(self, t: float) -> dict[str, int]:
        """Distributes my needs randomly over all my partners"""

        dist = dict()
        for needs, all_partners in [
            (self.awi.needed_supplies, self.awi.my_suppliers),
            (self.awi.needed_sales, self.awi.my_consumers),
        ]:
            # print(needs)
            # print(all_partners)
            # find suppliers and consumers still negotiating with me
            partners = [_ for _ in all_partners if _ in self.negotiators.keys()]
            n_partners = len(partners)
            # print(self.negotiators)
            # print(f"partners: {partners}, n_partners: {n_partners}")

            # if I need nothing, end all negotiations
            if (needs <= 0) or (n_partners == 0):
                dist.update(dict(zip(partners, [0] * n_partners)))
                continue

            # distribute my needs over my (remaining) partners.
            dist.update(
                dict(
                    zip(
                        partners,
                        distribute(
                            int(needs * (1 + self._overordering_fraction(t))),
                            n_partners,
                            equal=self.equal_distribution,
                            allow_zero=self.awi.allow_zero_quantity,
                        ),
                    )
                )
            )
        return dist
    
    def counter_all(self, offers, states):
        response = dict()
        future_partners = {
            k for k, v in offers.items() if v[TIME] != self.awi.current_step
        }
        offers = {k: v for k, v in offers.items() if v[TIME] == self.awi.current_step}
        # process for sales and supplies independently
        for needs, all_partners, issues in [
            (
                self.awi.needed_supplies,
                self.awi.my_suppliers,
                self.awi.current_input_issues,
            ),
            (
                self.awi.needed_sales,
                self.awi.my_consumers,
                self.awi.current_output_issues,
            ),
        ]:
            # get a random price
            price = issues[UNIT_PRICE].rand()
            # find active partners in some random order
            partners = [_ for _ in all_partners if _ in offers.keys()]
            random.shuffle(partners)
            partners = set(partners)

            # find the set of partners that gave me the best offer set
            # (i.e. total quantity nearest to my needs)
            plist = list(powerset(partners))[::-1]
            best_diff, best_indx = float("inf"), -1
            for i, partner_ids in enumerate(plist):
                offered = sum(offers[p][QUANTITY] for p in partner_ids)
                diff = abs(offered - needs)
                if diff < best_diff:
                    best_diff, best_indx = diff, i
                if diff == 0:
                    break
            unneeded_response = (
                SAOResponse(ResponseType.END_NEGOTIATION, None)
                if not self.awi.allow_zero_quantity
                else SAOResponse(
                    ResponseType.REJECT_OFFER, (0, self.awi.current_step, 0)
                )
            )

            # If the best combination of offers is good enough, accept them and end all
            # other negotiations
            th = self._allowed_mismatch(min(_.relative_time for _ in states.values()))
            if best_diff <= th:
                partner_ids = plist[best_indx]
                others = list(partners.difference(partner_ids).union(future_partners))
                response |= {
                    k: SAOResponse(ResponseType.ACCEPT_OFFER, offers[k])
                    for k in partner_ids
                } | {k: unneeded_response for k in others}
                continue

            # If I still do not have a good enough offer, distribute my current needs
            # randomly over my partners.
            t = min(_.relative_time for _ in states.values())
            distribution = self.distribute_needs(t)
            response.update(
                {
                    k: (
                        unneeded_response
                        if q == 0
                        else SAOResponse(
                            ResponseType.REJECT_OFFER, (q, self.awi.current_step, price)
                        )
                    )
                    for k, q in distribution.items()
                }
            )
        return response


class CustomActionWrapper(ActionWrapper):
    def __init__(self, env: OneShotEnv):
        super().__init__(env)
        self.env: OneShotEnv = env
        # self.supervisor_agent = CustomCautiousOneShotAgent()
        self.supervisor_agent = CustomEqualDistOneShotAgent()
        # self.supervisor_agent = CustomEpsilonGreedyAgent()

    def reset(self, *args, **kwargs):
        """Override the reset method to ensure proper initialization."""
        # print("CustomActionWrapper reset called")
        observation, info = self.env.reset(*args, **kwargs)
        # CustomCautiousOneShotAgent を再生成
        # self.supervisor_agent = CustomCautiousOneShotAgent()
        self.supervisor_agent = CustomEqualDistOneShotAgent()
        # self.supervisor_agent = CustomEpsilonGreedyAgent()
        return observation, info

    def process_offers(self, partners, responses):
        """
        Processes offers into groups and calculates the final array.

        Args:
            partners (list[str]): List of partner IDs.
            responses (dict[str, SAOResponse]): Dictionary of responses for each partner.

        Returns:
            np.ndarray: Array in the format [quantity, price, quantity, price, ...].
        """
        # グループ数を定義
        num_groups = 4

        # グループを初期化
        groups = [[] for _ in range(num_groups)]

        # パートナーをグループに分割
        for i, partner in enumerate(partners):
            groups[i % num_groups].append(partner)

        # 各グループの数量を足し合わせ、価格をランダムに決定
        result = []
        for group in groups:
            total_quantity = sum(
                responses[partner].outcome[0]
                for partner in group
                if partner in responses and responses[partner].outcome is not None
            )
            random_price = random.randint(0, 0)  # 価格をランダムに決定 (例: 10〜20の範囲)
            result.extend([total_quantity, random_price])

        return np.array(result)

    def action(self, action):
        """Override this method to modify the action before it is passed to the environment."""
        # Here you can modify the action as needed
        self.supervisor_agent.awi = self.env._agent.awi
        self.supervisor_agent.init()

        offers = self.env._agent.awi.current_offers
        states = self.env._agent.awi.current_states
        # print(f"offers: {self.env._agent.awi.current_offers}")
        # print(f"states: {self.env._agent.awi.current_states}")
        # print(f"my_suppliers: {self.env._agent.awi.my_suppliers}")
        # print(f"my_consumers: {self.env._agent.awi.my_consumers}")

        supervisor_offers = self.supervisor_agent.awi.current_offers
        supervisor_states = self.supervisor_agent.awi.current_states
        # print(f"Supervisor Offers: {supervisor_offers}")
        # print(f"Supervisor States: {supervisor_states}")
        # print(f"Supervisor Suppliers: {self.supervisor_agent.awi.my_suppliers}")
        # print(f"Supervisor Consumers: {self.supervisor_agent.awi.my_consumers}")
        

        response = self.supervisor_agent.counter_all(offers, states)
        # print(f"Response: {response}")


        if self.env._agent.awi.level == 0:
            result = self.process_offers(
                list(self.env._agent.awi.my_consumers),
                response
            )
        else:
            result = self.process_offers(
                list(self.env._agent.awi.my_suppliers),
                response
            )
        # print(f"Processed Action: {result}")

        # print(f"Wrapper_Action: {action}")

        action = result + action
        # print(f"Final Action: {action}")
        # input()



        return action
