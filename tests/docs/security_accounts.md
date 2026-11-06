# 認証・アカウント・安全性

ログイン、2FA、SSO、Bot対策、アカウント・暗号化を変更するとき。この表から関係するテストだけを選びます。通常の実装前にテスト本体を通読する必要はありません。

| テストファイル | 主な対象 |
|---|---|
| [test_2fa_success_animation_regressions.py](../test_2fa_success_animation_regressions.py) | 2fa / success / animation |
| [test_account_portability.py](../test_account_portability.py) | account / portability |
| [test_api_key_prompt_regressions.py](../test_api_key_prompt_regressions.py) | api / key / prompt |
| [test_bot_evidence_admin_regressions.py](../test_bot_evidence_admin_regressions.py) | bot / evidence / admin |
| [test_flask_login_flash_leak_regressions.py](../test_flask_login_flash_leak_regressions.py) | flask / login / flash / leak |
| [test_import_restore_regressions.py](../test_import_restore_regressions.py) | import / restore |
| [test_login_passkey_turnstile_regressions.py](../test_login_passkey_turnstile_regressions.py) | login / passkey / turnstile |
| [test_login_success_redirect_regressions.py](../test_login_success_redirect_regressions.py) | login / success / redirect |
| [test_minashin_sso_regressions.py](../test_minashin_sso_regressions.py) | minashin / sso |
| [test_oauth_login_success_animation_regressions.py](../test_oauth_login_success_animation_regressions.py) | oauth / login / success / animation |
| [test_security_regressions.py](../test_security_regressions.py) | security |
| [test_setup_import_regressions.py](../test_setup_import_regressions.py) | setup / import |
| [test_signup_session_regressions.py](../test_signup_session_regressions.py) | signup / session |
| [test_skills_api_regressions.py](../test_skills_api_regressions.py) | Skill API・旧Gem互換・gem→skill DB移行 |
| [test_turnstile_bot_detection_regressions.py](../test_turnstile_bot_detection_regressions.py) | turnstile / bot / detection |

実行・失敗時の手順は [workflow.md](workflow.md) を参照してください。
