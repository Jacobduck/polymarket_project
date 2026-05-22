"""polycluster - a pipeline for Polymarket insider-trading detection.

The package is organized along the steps of the pipeline:

* :mod:`polycluster.markets` - resolve a slug into a :class:`Market`
  carrying token IDs and a time window.
* :mod:`polycluster.events` - fetch OrderFilled events from the
  Polymarket orderbook subgraph for a market.
* :mod:`polycluster.parsing` - turn raw events into wallet-level trades
  and YES-price history dataframes.
* :mod:`polycluster.features` - compute per-(wallet, market) features
  capturing entry timing, holdings, PnL, markouts, and inventory.
* :mod:`polycluster.modeling` - assemble feature dataframes, impute,
  train logistic-regression classifiers, and explain predictions.
* :mod:`polycluster.viz` - plot a wallet's trades over price history.

Top-level imports re-export the public API for the steps that exist so
far. Helper functions are intentionally kept module-private and not
exposed here.
"""

from polycluster.events import (
    get_market_orderfilled_events,
    get_orderfilled_events,
    get_orderfilled_events_checkpointed,
)
from polycluster.features import (
    build_market_volume_df_from_orderfilled,
    compute_adaptive_markout_features,
    compute_contrarian_price_features,
    compute_entry_features,
    compute_extreme_inventory_and_day_features,
    compute_holding_features,
    compute_market_user_features,
    compute_markout_features,
    compute_markout_features1,
    compute_path_alignment_features,
    compute_realized_pnl_features,
    compute_realized_pnl_fifo,
    compute_unrealized_features,
    detect_major_move,
    get_user_features,
)
from polycluster.markets import Market, get_market_by_slug, get_markets_traded_by_wallet
from polycluster.modeling import (
    DEFAULT_BEHAVIORAL_FEATURES,
    build_feature_dataframe,
    explain_misclassification_direction,
    explain_single_prediction,
    fit_final_logistic,
    get_misclassified_rows,
    impute_feature_only_df,
    nested_kfold_loocv_logistic,
    run_loocv_elasticnet_logistic,
    run_loocv_logistic,
    score_new_rows,
    tune_C_loocv_logistic,
)
from polycluster.parsing import (
    build_history_df_from_orderfilled,
    get_user_trades_df,
    map_trades_to_history,
    normalize_trades_to_yes_df,
    normalize_trades_to_yes_dicts,
    parse_orderfilled_events,
)
from polycluster.viz import plot_user_on_history, plot_user_on_market

__all__ = [
    "DEFAULT_BEHAVIORAL_FEATURES",
    "Market",
    "build_feature_dataframe",
    "build_history_df_from_orderfilled",
    "build_market_volume_df_from_orderfilled",
    "compute_adaptive_markout_features",
    "compute_contrarian_price_features",
    "compute_entry_features",
    "compute_extreme_inventory_and_day_features",
    "compute_holding_features",
    "compute_market_user_features",
    "compute_markout_features",
    "compute_markout_features1",
    "compute_path_alignment_features",
    "compute_realized_pnl_features",
    "compute_realized_pnl_fifo",
    "compute_unrealized_features",
    "detect_major_move",
    "explain_misclassification_direction",
    "explain_single_prediction",
    "fit_final_logistic",
    "get_market_by_slug",
    "get_market_orderfilled_events",
    "get_markets_traded_by_wallet",
    "get_misclassified_rows",
    "get_orderfilled_events",
    "get_orderfilled_events_checkpointed",
    "get_user_features",
    "get_user_trades_df",
    "impute_feature_only_df",
    "map_trades_to_history",
    "nested_kfold_loocv_logistic",
    "normalize_trades_to_yes_df",
    "normalize_trades_to_yes_dicts",
    "parse_orderfilled_events",
    "plot_user_on_history",
    "plot_user_on_market",
    "run_loocv_elasticnet_logistic",
    "run_loocv_logistic",
    "score_new_rows",
    "tune_C_loocv_logistic",
]




"""polycluster is very slow and subgraph is not up to date. We also notice insiders trade towards the end of a market"""
"""test big market for goldsky limit"""
"""have a function to only plot price curve"""
"""0x5d0f03cf1243a3e21262d6cf844795afd9fff0ad"""
"""0x335592400e402c26583ce8b56d12605e9548a126"""
"""0x68558d37cafd9e6612ab32863f55ccdd798f655a"""
"""0xffa6b3c90514d7b861c87d7e51cc35fff34530fe"""
"""use LLM vision to determine if someone is an insider from a plot"""
"""do over-sampling"""
"""add artificial insiders to the dataset, then run the alg on polymarkets markets to find more insiders with LLM vision"""
"""try to find 50 insiders"""
"""find the law of the distribution of total PnL"""





"CAUGHT INSIDERS BELOW"



"""0x63c247f9e2722273120a2c4a4c0f01b658022c46""" #will aztec launch a token in 2025
"""0x0afc7ce56285bde1fbe3a75efaffdfc86d6530b2""" #israel strikes iran
"""0x8dc5fcff363eef19ad5f0bfce9f1fa216d456706""" #will-draftkings-launch-a-prediction-market-in-2025
"""0x55ea982cebff271722419595e0659ef297b48d7c""" #will-draftkings-launch-a-prediction-market-in-2025
"""0xbd810a483eb846da535405a442532f86dcfe1f7f""" ##will-draftkings-launch-a-prediction-market-in-2025
"""0x976685b6e867a0400085b1273309e84cd0fc627c""" #nothing-ever-happens-microstrategy #microstrategy-announces-1000-btc-purchase-december-2-8 #microstrategy-sell-any-bitcoin-in-2025 #will-microstrategy-announce-holding-650k-btc-by-november-30
"""0x5d55e3cf7a108e462186bedb04e090d4cd033bdf""" #will-microstrategy-announce-a-bitcoin-purchase-december-16-22 #microstrategy-announces-1000-btc-purchase-december-2-8
"""0x914e244ae32c19982d96ab50b3b55e487d1feace""" #nothing-ever-happens-microstrategy #will-microstrategy-announce-holding-740k-btc-by-march-31
"""0x567500e942fa8ea41bea3272014565d6466959bc""" #microstrategy-announces-1000-btc-purchase-november-11-17 #will-microstrategy-announce-holding-650k-btc-by-november-30 #microstrategy-announces-1000-btc-purchase-december-2-8
"""0x6e1a7bd753e97eaa367f45f4229d82f176475633""" #microstrategy-announces-1000-btc-purchase-december-2-8 #will-microstrategy-purchase-bitcoin-july-1-7
"""0xaab69d14e74adf2e46459af9a6b512dbedd2f297""" #will-microstrategy-announce-a-bitcoin-purchase-december-16-22 #microstrategy-announces-1000-btc-purchase-december-2-8 #will-microstrategy-purchase-bitcoin-july-1-7 #will-microstrategy-hold-620k-btc-before-august
"""0x55ac0f2ea2aa935b88385fe4adb98dd3a60f1023""" #will-microstrategy-announce-a-bitcoin-purchase-december-16-22 #will-microstrategy-hold-680k-btc-by-december-31
"""0x0d832e843cd972ffc9549ed1b6cd6aabf1c49fc7""" #microstrategy-announces-1000-btc-purchase-december-2-8 
"""0x185ffa4bd5bebfd8a463b905137ebd801bb7053c""" #will-stable-launch-a-token-in-2025
"""0x4b7e367dd40de1d629a09bc414ec7b14e97c8736""" #will-stable-launch-a-token-in-2025
"""0xe3a1bd7fa34f49c5e11eb20787400090d9cd1ede""" #will-lighter-perform-an-airdrop-by-december-31 #will-stable-launch-a-token-in-2025
"""0xc8809a2756d80cbbf5c81a685f2ce113b3242c56""" #will-stable-launch-a-token-in-2025
"""0xa430506774f9efaf39903ee7e0db1351f66891ca""" #will-yulia-navalnaya-win-the-nobel-peace-prize-in-2025 #will-mara-corina-machado-win-the-nobel-peace-prize-in-2025
"""0x234cc49e43dff8b3207bbd3a8a2579f339cb9867""" "this guy made 1 prediction in his life and won" #will-mara-corina-machado-win-the-nobel-peace-prize-in-2025 
"""0x1d9af60c679cd0b577c3c4ccb4b1a4be4174426d""" #will-axiom-be-accused-of-insider-trading 
"""0xe56526b27b96f009b31ddb46558a134047bfce48""" #will-axiom-be-accused-of-insider-trading
"""0x054ec2f0ccfdae941886a3ed306635068c716639""" #will-axiom-be-accused-of-insider-trading
"""0x6d6affce1ed04a0e9611484daf1cef5cbcf3fb40""" #will-axiom-be-accused-of-insider-trading
"""0x581f34349babaf03b2d3c8f5f60cf44ffbe19a3a""" #will-axiom-be-accused-of-insider-trading
"""0xaab29084bcc42daff9e11b4a5a4cc55cda3eb306""" #will-axiom-be-accused-of-insider-trading
"""0x98a96619e482700e83e8486e4f3727dba17f5381""" #will-axiom-be-accused-of-insider-trading

"""https://polymarket.com/0x8a480B60Cb78213fDf34842160dc956a831a2558""" #many USA attack on Iran markets. All these guys have joined feb 2026, 3 weeks before attack on iran
"""https://polymarket.com/@flipfloppity""" #same as above
"""0x95bd107886bd48d6737029603e3db242fb781652""" #us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x5e0b017aef88af84fb3897fecce7413c6754981a""" #us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x344c5073e9b152a95d9f42767ac084a87fad68ab""" #us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x943117ea8e3f5b160e158c1c4b1786fee5d3e5a0""" #us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0xf37af11e538c6eba3ddcea540ee54ca35c6354f8""" #us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x45f402894f550357ff1d5db385c93bb9e480dd79"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x4ec2f11d2893eb1bc3bab7449910540c93adf7ec"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0xb390ae2cc3fd8cb2670a62129faeb775faee106c"""#khamenei-out-as-supreme-leader-of-iran-by-march-31 #us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x3e545810f8b95dd0fcf4114b0246a0dbb40ad1c5""" #khamenei-out-as-supreme-leader-of-iran-by-march-31 #us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0xa34d11f372e8a87acc87e3beabb32e2fe7f9ee4d"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0xb7200eb4ffc3ea4bde0ef28110238bd9f539c414"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x421b26a47bb5136d045190d7870095eb29498469"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0xab54f4b400bd8015e206a74f8d89362cf8eee856"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0xfffee3116390fc9415956617ed048d26c7c1b424"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0xdad42409cf398c9149c50217e4dc4a039d9dec0b"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0xe1bdfa52bc9342115bbf1d41d04b8cf0bcd8f5a7"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0xeee545e9706a03e73c5d17632e2bbb6f1096cce1"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x65387d3235b2fd6ac3587df1fff2c2de2f98e523"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0xe85f1414171cd41707dc7b256bd3db269f94a28c"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x244cde3e010a8eb27f0ff588476091bd3d73f1dc"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x62730fede68ebedfb510e6f38da2aa2eca081741"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0xcb8ed92b702a0f2397f8cda9e4925867b7b9135e"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0xf1d54d55b7568e6e4c20867ccf377555cd947ffc"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x6130238e132bd12ba92244568815c2f7fe4fc5c1"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0xeb1cdc2f267723b577cd0d1441fe5db05d440f79"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0xf1a1bcf248a0dc05bec474c229f490db00106f24"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x8e588259e06da665020dac0e4958301a2089579e"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x1270fccb862f069b80181dcedf7be31ec6cfc2c8"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x22d0803925702b81d038c9b3f3aa77a3727f218f"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x79963fd69baccad9bb8ab51ddfbf1f13e0630e13"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x82ed2416f0a49cc02fb529b00c03582598d17cc3"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0xc0292a841a0c9a7320aa39075cffcf1b8f64f705"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0xdcce3bfa065cf5ca9a0acfe83d6ecd8279d10a15"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x2b6072aa2255e3a95154180076771417e3dc0b9e"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766
"""0x26ecf4f4a01bf39ce98144ef9aeddf28e9548f54"""#us-strikes-iran-by-february-28-2026-227-967-547-688-589-491-592-418-452-924-384-915-464-672-196-157-993-596-269-535-381-391-471-256-988-997-296-225-762-973-292-827-345-182-558-215-794-879-189-761 #us-strikes-iran-by-march-1-2026-492 #will-us-or-israel-strike-iran-by-february-28-2026-766

"""0xe4b2396323001f880d1429400f5a8259845e5dc9""" #will-openai-release-a-new-frontier-model-on-december-11 #will-openai-release-a-new-frontier-model-by-december-13-683
"""0x7868856f93438c59a0b052161098a27768a65fc0""" #openai-browser-by-october-31 #openai-social-app-in-2025 #openai-browser-in-2025
"""0x40d9ac81a425f14d2c490c41ac8969c0cbcfd472""" #will-karol-g-perform-during-the-super-bowl-lx-halftime-show #will-cardi-b-perform-during-the-super-bowl-lx-halftime-show

"""0x8039ad26298d7847799899808554474b7fa57421""" #us-x-iran-ceasefire-by-april-7
"""0x755519c3a4a69469f488197fbd39b12f70b3ecc5""" #us-x-iran-ceasefire-by-april-7
"""0xd9875d4a0573dd3890738aab990938a53c360041""" #us-x-iran-ceasefire-by-april-7
"""0x68558d37cafd9e6612ab32863f55ccdd798f655a""" #us-x-iran-ceasefire-by-april-7



"""TOTAL: 174 (wallet, market slug) pairs across 77 unique wallets and 37 unique markets"""


""" We set scale_pos_weight = n_negatives / n_positives ≈ 6.3, 
which tells the gradient-boosted trees: "treat each missed insider as 6.3x worse than a false alarm." 
For precision focus, we'll then lower this weight (so the model is less eager to call someone an insider) 
and additionally raise the decision threshold above 0.5 at inference time — both lean the model 
toward "only flag if confident."""



"""faire en sorte que le training set soit équilibré, mais pas le test set"""
"""build a better model and iterate"""
"""feature: similarity between markets"""
"""precision (no false accusations) or recall (catch everyone)"""