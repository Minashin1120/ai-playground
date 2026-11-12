# --- Background: E2EE migration task ---

def migrate_e2ee_task(user_id, target_enable):
    with app.app_context():
        r = redis.from_url(REDIS_URL)
        r.set(f"migration_status:{user_id}", "processing")
        try:
            user = User.query.get(user_id)
            if not user: return
            # Estimate total work units (messages + files)
            total = 0
            done = 0
            threads = Thread.query.filter_by(user_id=user_id).all()
            total += sum(len(t.messages) for t in threads)
            user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
            if os.path.exists(user_dir):
                for root, _, files in os.walk(user_dir):
                    total += len(files)
            if total <= 0: total = 1
            r.set(f"migration_progress:{user_id}", f"{done}/{total}")
            user.enable_e2ee = target_enable
            if user.system_prompt:
                if target_enable: user.system_prompt = encrypt_val(user.system_prompt)
                else: user.system_prompt = decrypt_val(user.system_prompt)
            for t in threads:
                for m in t.messages:
                    if m.content:
                        if target_enable and not m.is_encrypted: m.content = encrypt_val(m.content)
                        elif not target_enable and m.is_encrypted: m.content = decrypt_val(m.content)
                    if m.thought_data:
                        if target_enable and not m.is_encrypted: m.thought_data = encrypt_val(m.thought_data)
                        elif not target_enable and m.is_encrypted: m.thought_data = decrypt_val(m.thought_data)
                    m.is_encrypted = target_enable
                    done += 1
                    if done % 10 == 0:
                        r.set(f"migration_progress:{user_id}", f"{done}/{total}")
            user_dir = os.path.join(app.config['UPLOAD_FOLDER'], str(user_id))
            if os.path.exists(user_dir):
                for root, dirs, files in os.walk(user_dir):
                    for file in files:
                        fp = os.path.join(root, file)
                        if target_enable:
                            if not file.endswith('.enc'):
                                with open(fp, 'rb') as f: data = f.read()
                                with open(fp + '.enc', 'wb') as f: f.write(encrypt_bytes(data))
                                secure_delete(fp)
                        else:
                            if file.endswith('.enc'):
                                with open(fp, 'rb') as f: data = decrypt_bytes(f.read())
                                new_fp = fp[:-4]
                                with open(new_fp, 'wb') as f: f.write(data)
                                secure_delete(fp)
                        done += 1
                        if done % 5 == 0:
                            r.set(f"migration_progress:{user_id}", f"{done}/{total}")
            safe_db_commit()
            r.set(f"migration_progress:{user_id}", f"{total}/{total}")
            r.set(f"migration_status:{user_id}", "done")
        except Exception as e:
            logger.error(f"Migration failed: {e}")
            r.set(f"migration_status:{user_id}", "error")
            r.set(f"migration_progress:{user_id}", "error")

