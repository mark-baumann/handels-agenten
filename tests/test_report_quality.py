from tradingagents.report_quality import (
    analyze_report_quality,
    evaluate_final_decision,
    resolve_agent_conflict,
)


def test_report_quality_flags_conflicting_signals():
    report = """
    Market Analyst Report
    FINAL TRANSACTION PROPOSAL: **HOLD**

    Trader Agent Report
    FINAL TRANSACTION PROPOSAL: **BUY**

    Research Manager
    Overweight
    """

    result = analyze_report_quality(report)

    assert result["is_valid"] is False
    assert any("conflicting" in issue.lower() for issue in result["issues"])


def test_report_quality_flags_missing_sections():
    report = """
    Market Analyst Report
    FINAL TRANSACTION PROPOSAL: **BUY**

    Research Manager
    **Recommendation**: Buy
    """

    result = analyze_report_quality(report)

    assert result["is_valid"] is False
    assert any("sentiment" in issue.lower() for issue in result["issues"])
    assert any("news" in issue.lower() for issue in result["issues"])
    assert any("fundamentals" in issue.lower() for issue in result["issues"])


def test_resolve_agent_conflict_downgrades_to_hold_when_reports_conflict():
    report = """
    Market Analyst Report
    FINAL TRANSACTION PROPOSAL: **HOLD**

    Trader Agent Report
    FINAL TRANSACTION PROPOSAL: **BUY**

    Research Manager
    **Recommendation**: Overweight
    """

    result = resolve_agent_conflict(report)

    assert result["accepted"] is False
    assert result["final_action"] == "HOLD"
    assert "position_size" in result


def test_evaluate_final_decision_rejects_invalid_buy_signal():
    report = """
    Market Analyst Report
    FINAL TRANSACTION PROPOSAL: **BUY**

    Trader Agent Report
    FINAL TRANSACTION PROPOSAL: **BUY**
    """

    result = evaluate_final_decision(report, "BUY")

    assert result["status"] == "rejected"
    assert result["final_action"] == "HOLD"
    assert result["position_size"] == "0-5%"
