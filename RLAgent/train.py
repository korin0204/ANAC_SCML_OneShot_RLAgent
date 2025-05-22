# trains an RL model
#
import logging
from typing import Any

from negmas.sao import SAOResponse
from rich import print
from scml.oneshot.awi import OneShotAWI
from scml.oneshot.rl.action import FlexibleActionManager
from scml.oneshot.rl.observation import FlexibleObservationManager
from scml.oneshot.rl.agent import OneShotRLAgent
from scml.oneshot.rl.common import model_wrapper
from scml.oneshot.rl.env import OneShotEnv
from scml.oneshot.rl.reward import DefaultRewardFunction

# sys.path.append(str(Path(__file__).parent))
from .common import MODEL_PATH, MyObservationManager, TrainingAlgorithm, make_context

from scml.oneshot.context import GeneralContext, SupplierContext, ConsumerContext
from .mycontext import MySupplierContext, MyConsumerContext
# from .myaction import MyActionManager
# from .myrlagent import MyOneShotRLAgent, set_negotiation_failure, set_negotiation_success, get_negotiation_failure, get_negotiation_success

import numpy as np

NTRAINING = 10000  # number of training steps /1000~5000steps per episode
NADDTRAINING = 10000  # number of training steps /1000~5000steps per episode

import os
import csv
TEMP_LOG_PATH = "oneshot_rl/myagent/train_log/profit.csv"

from itertools import chain, combinations

def powerset(offer_list):
    """
    Generate the powerset (all subsets) of the given list.
    
    Args:
        offer_list (list): A list of offers (e.g., [(4, 15), (3, 20)]).
    
    Returns:
        list: A list of all subsets of the input list.
    """
    # chain.from_iterableを使ってすべての部分集合を生成
    return list(chain.from_iterable(combinations(offer_list, r) for r in range(len(offer_list) + 1)))

def offer_value(offer_powerset_list):
    '''冪集合から各部分集合の量と単価を掛けたものを合計を計算する関数'''
    # 各部分集合の合計を計算
    value_sums = []  # 量 × 単価の合計
    quantity_sums = []  # 量の合計
    for subset in offer_powerset_list:
        total_value = sum(quantity * price for quantity, price in subset)
        total_quantity = sum(quantity for quantity, price in subset)
        value_sums.append(total_value)
        quantity_sums.append(total_quantity)
    
    return value_sums, quantity_sums

def negotiation_details(awi: OneShotAWI):
    
    # awi.current_negotiation_detailsの中身を確認
    for key, value in awi.current_negotiation_details.items():
        current_proposer = [] # 誰からの提案か
        new_offerer_agents = [] # 誰に提案したか
        offered_list = []  # 提案された内容のリストを初期化
        offer_list = [] # 提案した内容のリストを初期化
        is_current_proposer_me = False # 自分が提案者か？
        # print(f"Key: {key}, Value: {value}")

        # "buy"または"sell"キーの中に交渉の詳細がある場合
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                # print(f"  Sub-Key: {sub_key}, Sub-Value: {sub_value}")

                # NegotiationDetailsオブジェクトの中にnmiがある場合
                if hasattr(sub_value, "nmi"):
                    nmi = sub_value.nmi
                    if hasattr(nmi, "_mechanism"):
                        mechanism = nmi._mechanism
                        if hasattr(mechanism, "_history"):
                            history = mechanism._history
                            # print(f"    History for {sub_key}:")
                            # for i, state in enumerate(history):
                            #     print(f"      State {i + 1}:")
                            #     print(f"        Running: {state.running}")
                            #     print(f"        Step: {state.step}")
                                
                                # print(f"        Time: {state.time}")
                                # print(f"        Relative Time: {state.relative_time}")
                                
                                # print(f"        Current Offer: {state.current_offer}")
                                # print(f"        Current Proposer: {state.current_proposer}")
                                # print(f"        New Offers: {state.new_offers}")
                                # print(f"        Agreement: {state.agreement}")
                                
                                # print(f"        Timed Out: {state.timedout}")
                                # print(f"        Broken: {state.broken}")
                                # print()
                                # print("-" * 50)
                                
                    

                        # SAONMIオブジェクトのcurrent_stateを取得
                        if hasattr(mechanism, "_current_state"):
                            current_state = mechanism._current_state
                            # print(f"    Current State for {sub_key}:")                              
                            # print(f"      Running: {current_state.running}")
                            # print(f"      Step: {current_state.step}")
                            # print(f"      Current Offer: {current_state.current_offer}")
                            
                            # if current_state.current_offer is not None:
                            #     offer_list.append((current_state.current_offer[0], current_state.current_offer[2]))
                            
                            
                            # print(f"      Current Proposer: {current_state.current_proposer}")
                            
                            if current_state.current_proposer is not None:
                                if str(current_state.current_proposer) is not str(awi.agent): # offerが自分から出ない場合
                                    current_proposer.append(current_state.current_proposer) # 誰からの提案か
                                    if current_state.current_offer is not None:
                                        offered_list.append((current_state.current_offer[0], current_state.current_offer[2])) # このoffered_listに追加するのは受け取ったofferのみ
                                else: # 自分が提案者の場合
                                    new_offerer_agents.append(current_state.new_offerer_agents[0]) # 誰への提案か
                                    if current_state.current_offer is not None:
                                        offer_list.append((current_state.current_offer[0], current_state.current_offer[2])) # offer_listに追加するのは自分が提案したofferのみ
                                        
                            # print(f"      New Offers: {current_state.new_offers}")
                            # print(f"      Agreement: {current_state.agreement}")
                            # print(f"      Timed Out: {current_state.timedout}")
                            # print(f"      Broken: {current_state.broken}")
                            # print("-" * 50)
                        
        # print(f"Total Offer Quantity: {total_offer_quantity}") # 交渉中の合計数量(offerもcounterも含む)これが自分のrequireとかけ離れていると損
        # print(f"Total Estimated Price: {total_estimated_price}") # 取引量に価格もかけて、どれくらいの販売益が見込めるか(ただしペナルティは考慮していない)
    return current_proposer, new_offerer_agents, offered_list, offer_list

class MyRewardFunction(DefaultRewardFunction):
    max_estimated_profit_history = [] # 見込み利益の履歴を保存するリスト

    """Custom reward function that considers profit and successful contracts"""
    def before_action(self, awi: OneShotAWI) -> dict:
        """
        Save the agent's balance and the number of successful contracts before the action.
        """
        """def and print market info"""
        # print("\n----------before_action----------")
        ex_my_input = awi.current_exogenous_input_quantity # 外生契約の入力量(自分のみ)
        ex_my_output = awi.current_exogenous_output_quantity # 外生契約の出力量(自分のみ)
        ex_input = awi.exogenous_contract_summary[0][0] # 外生契約の入力量(総合)
        ex_output = awi.exogenous_contract_summary[2][0] # 外生契約の出力量(総合)
        current_exogenous_input_price = awi.current_exogenous_input_price
        current_exogenous_output_price = awi.current_exogenous_output_price
        level = awi.level # エージェントのレベルに応じて
        current_disposal_cost = awi.current_disposal_cost # 廃棄コスト
        current_shortfall_penalty = awi.current_shortfall_penalty # 不足ペナルティ
        current_balance = awi.current_balance # 現在のバランス(資金)
        current_score = awi.current_score # 現在のスコア(利益)
        current_step = awi.current_step # 現在のステップ
        needed_sales = awi.needed_sales # あと売らなければいけない量
        needed_supplies = awi.needed_supplies

        # current_proposer, new_offerer_agents, offered_list, offer_list = negotiation_details(awi) # 現在の交渉の詳細を表示して重要な値を返す
        """offer_listの冪集合を計算し、offerの組み合わせで最も価値が高いものを計算"""
        # offered_powerset_list = powerset(offered_list) # 提案の冪集合を計算
        # offered_total_value_list, offered_total_quantity_list = offer_value(offered_powerset_list) # 提案の冪集合から各部分集合の量と単価を掛けたものを合計を計算

        return {
            "balance": current_balance,  # 現在のバランス(資金)を記録
            "score": current_score,  # 現在のスコア(利益)を記録
            "current_step": current_step,  # 現在のステップを記録
            # "max_estimated_profit": max_estimated_profit,  # 見込み利益の最大を記録
            "needed_sales": needed_sales,  # あと売らなければいけない量を記録
            "needed_supplies": needed_supplies,  # あと仕入れなければいけない量を記録
        }




    def __call__(self, awi: OneShotAWI, action: dict[str, SAOResponse], info: dict) -> float:
        """
        Calculate the reward based on profit and successful contracts.
        """
        """field of init var"""
        reward = 0.0
        """def and print market info"""
        ex_my_input = awi.current_exogenous_input_quantity # 外生契約の入力量(自分のみ)
        ex_my_output = awi.current_exogenous_output_quantity # 外生契約の出力量(自分のみ)
        ex_input = awi.exogenous_contract_summary[0][0] # 外生契約の入力量(総合)
        ex_output = awi.exogenous_contract_summary[2][0] # 外生契約の出力量(総合)
        current_exogenous_input_price = awi.current_exogenous_input_price
        current_exogenous_output_price = awi.current_exogenous_output_price
        level = awi.level # エージェントのレベルに応じて
        current_disposal_cost = awi.current_disposal_cost # 廃棄コスト
        current_shortfall_penalty = awi.current_shortfall_penalty # 不足ペナルティ
        current_balance = awi.current_balance # 現在のバランス(資金)
        current_score = awi.current_score # 現在のスコア(利益)
        current_step = awi.current_step # 現在のステップ
        needed_sales = awi.needed_sales # あと売らなければいけない量
        needed_supplies = awi.needed_supplies
        
        # current_proposer, new_offerer_agents, offered_list, offer_list = negotiation_details(awi) # 現在の交渉の詳細を表示して重要な値を返す
        # print(f"current_proposer: {current_proposer}, me: {awi.agent}") # 現在の提案者を表示
        
        """offer_listの冪集合を計算し、offerの組み合わせで最も価値が高いものを計算"""
        # offered_powerset_list = powerset(offered_list) # 提案の冪集合を計算
        # offered_total_value_list, offered_total_quantity_list = offer_value(offered_powerset_list) # 提案の冪集合から各部分集合の量と単価を掛けたものを合計を計算
        # print(f"offered_total_value_list: {offered_total_value_list}") # 提案の合計を表示
        # print(f"offered_total_quantity_list: {offered_total_quantity_list}") # 提案の合計を表示

        """受け入れ戦略(相手の受け入れも加味)"""
        # あと売ら/仕入れなければいけない量が少なくなれば良い行動として報酬を与える
        if (current_step == info["current_step"]): #action前が同じ行動であるならば
            if (level == 0):
                if(needed_sales > -1): # 0, 1, 2, ...
                    """金額ベース報酬"""
                    # reward += (awi.trading_prices[1] - 1) * (info["needed_sales"] - needed_sales) # 販売の利益 l0 安い方で見積もる
                    # reward += (current_disposal_cost) * (info["needed_sales"] - needed_sales) # 支払う必要のなくなった廃棄コスト
                    """回数ベース報酬"""
                    if (info["needed_sales"] - needed_sales) > 0: # 残りの数に変化があれば
                        reward += (awi.trading_prices[1] - 1) * (info["needed_sales"] - needed_sales)
                else:
                    """金額ベース報酬"""
                    # if info["needed_sales"] > -1:
                        # reward += (awi.trading_prices[1] - 1) * (info["needed_sales"] - 0) # 販売の利益 l0 安い方で見積もる
                        # reward += (current_disposal_cost) * (info["needed_sales"] - 0) # 支払う必要のなくなった廃棄コスト
                        # reward -= (current_shortfall_penalty) * (0 - needed_sales) # 多く売りすぎてしまったときに発生する不足ペナルティ
                    """回数ベース報酬"""
                    if (info["needed_sales"] - needed_sales) > 0: # 残りの数に変化があれば
                        reward -= (current_shortfall_penalty) * (0 - needed_sales)

            else:
                if(needed_supplies > -1):
                    """金額ベース報酬"""
                    # reward += (awi.trading_prices[2] - 1) * (info["needed_supplies"] - needed_supplies) # 最終製品を売ることで得られる利益
                    # reward += (current_shortfall_penalty) * (info["needed_supplies"] - needed_supplies) # 商品を仕入れることで払う必要のなくなった不足ペナルティの量 l1
                    """回数ベース報酬"""
                    if (info["needed_supplies"] - needed_supplies) > 0: # 残りの数に変化があれば
                        reward += (awi.trading_prices[2] - 1) * (info["needed_supplies"] - needed_supplies)
                else:
                    """金額ベース報酬"""
                    # if info["needed_supplies"] > -1:
                        # reward += (awi.trading_prices[2] - 1) * (info["needed_supplies"] - 0) # 最終製品を売ることで得られる利益
                        # reward += (current_shortfall_penalty) * (info["needed_supplies"] - 0) # 商品を仕入れることで払う必要のなくなった不足ペナルティの量 l1
                        # reward -= (current_disposal_cost) * (0 - needed_supplies) # 多く仕入れすぎたことによって発生する廃棄コスト
                    """回数ベース報酬"""
                    if (info["needed_supplies"] - needed_supplies) > 0:
                        reward -= (current_disposal_cost) * (0 - needed_supplies)
        
        # reward *= 0.05 # 報酬を0.1倍することで、報酬のスケールを調整する

        # 所持金の増加量を計算
        profit = current_balance - info["balance"] # -500~500程度
        # print(f"profit: {profit}") # 所持金の変化を表示
        reward += profit # 所持金の変化を報酬に加算
        # offer(量, 時間(無視される), 単価)
        # if profit > 0:
        #     reward += 1
        # else:
        #     reward -= 1

        # # 数量部分を加算
        # total_offer_quantity = sum(quantity for quantity, _ in offer_list) # 提案した数量
        # total_offered_quantity = sum(quantity for quantity, _ in offered_list) # 提案された数量


        """提案戦略"""
        # if (level == 0) and (total_offered_quantity < needed_sales): # もらった提案が必要数に満たない場合
        #     if(total_offer_quantity == (needed_sales - total_offered_quantity)): # 自分の提案で必要数を満たすようにしている場合
        #         reward += 1
        #     else: # 自分の提案で必要数を満たしていない場合
        #         None
        # elif (level == 1) and (total_offered_quantity < needed_supplies): # もらった提案が必要数に満たない場合
        #     if(total_offer_quantity == (needed_supplies - total_offered_quantity)): # 自分の提案で必要数を満たすようにしている場合
        #         reward += 1
        #     else: # 自分の提案で必要数を満たしていない場合
        #         None
        
        # print(f"reward: {reward}") # 報酬を表示
        # 正則化
        reward = np.tanh(reward/30) # 報酬を正則化する
        return reward


# OneShotEnvが定義する行動空間や観測や報酬など様々なことについてmake_envで設定する
def make_env(as_supplier, log: bool = False) -> OneShotEnv:
    log_params: dict[str, Any] = (
        dict(
            no_logs=False,
            log_stats_every=1,
            log_file_level=logging.DEBUG,
            log_screen_level=logging.ERROR,
            save_signed_contracts=True,
            save_cancelled_contracts=True,
            save_negotiations=True,
            save_resolved_breaches=True,
            save_unresolved_breaches=True,
            debug=True,
        )
        if log
        else dict(debug=True)
    )
    log_params.update(
        dict(
            ignore_agent_exceptions=False,
            ignore_negotiation_exceptions=False,
            ignore_contract_execution_exceptions=False,
            ignore_simulation_exceptions=False,
        )
    )
    # context = make_context(as_supplier, strength)
    context = MySupplierContext() if as_supplier else MyConsumerContext()
    # context = SupplierContext() if as_supplier else ConsumerContext()
    return OneShotEnv(
        action_manager=FlexibleActionManager(context=context), # Agentの行動を管理 行動空間(提示価格や数量,交渉の拒否)をA2Cに渡す
        # action_manager=MyActionManager(context=context),  # type: ignore # Agentの行動を管理 行動空間(提示価格や数量,交渉の拒否)をA2Cに渡す
        observation_manager=MyObservationManager(context=context),  # type: ignore # 環境の観測者
        # observation_manager=FlexibleObservationManager(context=context),  # type: ignore # 環境の観測者
        reward_function=MyRewardFunction(), # 報酬関数を指定
        context=context, # 環境のコンテキストを指定
        extra_checks=False,
    )


def try_a_model(
    model,
    as_supplier: bool,
    strength: int,
):
    """Runs a single simulation with one agent controlled with the given model"""

    obs_type = MyObservationManager
    # obs_type = FlexibleObservationManager
    # Create a world context compatibly with the model
    # context = make_context(as_supplier, strength)
    context = MySupplierContext() if as_supplier else MyConsumerContext()
    # context = SupplierContext() if as_supplier else ConsumerContext()
    # sample a world and the RL agents (always one in this case)
    world, _ = context.generate(
        types=(OneShotRLAgent,),
        params=(
            dict(
                models=[model_wrapper(model)],
                observation_managers=[obs_type(context)],
                action_managers=[FlexibleActionManager(context)],
                # action_managers=[MyActionManager(context)],
            ),
        ),
    )
    # run the world simulation
    world.run_with_progress()
    return world


def main(ntrain: int = NTRAINING):
    # choose the type of the model. Possibilities supported are:
    # fixed: Supports a single world configuration
    # limited: Supports a limited range of world configuration
    # unlimited: Supports any range of world configurations
    test_only = True
    if test_only:
        as_supplier = False
        strength=2
        # 4-8
        model_path = (
            MODEL_PATH.parent
            / f"{MODEL_PATH.name}{'_supplier' if as_supplier else '_consumer'}{'_4-8_PPO_tanh_double_300000'}"
        )
        
        # 4-4
        # model_path = (
        #     MODEL_PATH.parent
        #     / f"{MODEL_PATH.name}{'_supplier' if as_supplier else '_consumer'}{'_add_learn_2000000'}"
        # )


        # なぜかlearnをはさむとテスト時のエージェントの位置が変わる
        env = make_env(as_supplier, strength)
        model = TrainingAlgorithm(  # type: ignore learning_rate must be passed by the algorithm itself
                        "MlpPolicy", env, verbose=1, 
                        # tensorboard_log="./learn_log/"
                    )

        # train the model
        model.learn(total_timesteps=1,
                    progress_bar=False,
                    log_interval=100,
                    )
        # model.save(model_path)
        del model



        print("評価をする")
        model = TrainingAlgorithm.load(model_path)
        # try the model in a single simulation
        world = try_a_model(model, as_supplier, strength)
        print(world.scores())

    else:
        for as_supplier in (True, False):
            # for strength in (-1, 0, 1):
                strength=2
                add_learn = True # you can set this to True to add learn
                if add_learn:
                    model_path = MODEL_PATH.parent / f"{MODEL_PATH.name}{'_supplier' if as_supplier else '_consumer'}{'_4-8_A2C_tanh_'}{'640000'}"
                    model = TrainingAlgorithm.load(model_path)
                    env = make_env(as_supplier)
                    model.set_env(env)
                    model.learn(total_timesteps=NADDTRAINING, progress_bar=True, log_interval=100)
                    model_path = MODEL_PATH.parent / f"{MODEL_PATH.name}{'_supplier' if as_supplier else '_consumer'}{'_4-8_A2C_tanh_'}{640000 + NADDTRAINING}"
                else:
                    # create a gymnasium environment for training
                    env = make_env(as_supplier)

                    # choose a training algorithm
                    # 引数のenvによって行動空間や観測空間が決まる
                    # それをもとにA2Cが学習する
                    # A2Cの場合
                    model = TrainingAlgorithm(  # type: ignore learning_rate must be passed by the algorithm itself
                        "MlpPolicy", 
                        env, 
                        verbose=1, 
                        tensorboard_log="./learn_log/"
                    )

                    # train the model
                    model.learn(total_timesteps=ntrain,
                                progress_bar=True,
                                log_interval=100,
                                )
                    print(
                        f"\tFinished training the model for {ntrain} steps ... Testing it on a single world simulation"
                    )

                    # decide the model path to save to
                    model_path = (
                        MODEL_PATH.parent
                        / f"{MODEL_PATH.name}{'_supplier' if as_supplier else '_consumer'}{'_4-8_PPO_tanh_double_'}{NTRAINING}"
                    )

                # save the model
                model.save(model_path)
                # remove the in-memory model
                del model
                # load the model
                print("評価をします")
                model = TrainingAlgorithm.load(model_path)
                # try the model in a single simulation
                world = try_a_model(model, as_supplier, strength)
                print(world.scores())


if __name__ == "__main__":
    import sys

    main(int(sys.argv[1]) if len(sys.argv) > 1 else NTRAINING)