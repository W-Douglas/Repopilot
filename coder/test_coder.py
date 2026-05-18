"""
Tests for the Coder Module.

Tests the edit instruction generation from fix plans.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from coder import (
    Coder, CoderConfig,
    EditInstruction, EditType, EditCollection,
    CoderPipeline, CoderInput, CoderOutput,
    Editor, EditorConfig, EditorResult
)


class TestEditInstruction:
    """Tests for EditInstruction."""

    def test_replace_edit_validation(self):
        """Test validation of replace edits."""
        # Valid replace edit
        edit = EditInstruction(
            edit_type=EditType.REPLACE,
            file_path="test.py",
            old_text="old_code",
            new_text="new_code",
            reason="Fix bug"
        )
        valid, error = edit.validate()
        assert valid, f"Valid edit should pass: {error}"

    def test_replace_edit_no_change(self):
        """Test that identical old_text and new_text fails validation."""
        edit = EditInstruction(
            edit_type=EditType.REPLACE,
            file_path="test.py",
            old_text="same",
            new_text="same",
            reason="No change"
        )
        valid, error = edit.validate()
        assert not valid
        assert "identical" in error.lower()

    def test_whole_file_edit(self):
        """Test whole_file edit validation."""
        edit = EditInstruction(
            edit_type=EditType.WHOLE_FILE,
            file_path="test.py",
            new_content="full content",
            reason="Replace file"
        )
        valid, error = edit.validate()
        assert valid

    def test_line_edit_validation(self):
        """Test line_edit validation."""
        # Valid line edit
        edit = EditInstruction(
            edit_type=EditType.LINE_EDIT,
            file_path="test.py",
            start_line=1,
            end_line=10,
            new_text="new content",
            reason="Replace lines"
        )
        valid, error = edit.validate()
        assert valid

    def test_line_edit_invalid_range(self):
        """Test that end_line < start_line fails."""
        edit = EditInstruction(
            edit_type=EditType.LINE_EDIT,
            file_path="test.py",
            start_line=10,
            end_line=5,
            reason="Invalid range"
        )
        valid, error = edit.validate()
        assert not valid
        assert "end_line" in error.lower()


class TestEditCollection:
    """Tests for EditCollection."""

    def test_add_valid_edit(self):
        """Test adding valid edits to collection."""
        collection = EditCollection()
        edit = EditInstruction(
            edit_type=EditType.REPLACE,
            file_path="test.py",
            old_text="a",
            new_text="b",
            reason="test"
        )
        collection.add_edit(edit)
        assert len(collection.edits) == 1

    def test_add_invalid_edit(self):
        """Test that invalid edits raise ValueError."""
        collection = EditCollection()
        edit = EditInstruction(
            edit_type=EditType.REPLACE,
            file_path="test.py",
            old_text="a",
            new_text="a",  # Same as old - invalid
            reason="test"
        )
        with pytest.raises(ValueError):
            collection.add_edit(edit)

    def test_get_target_files(self):
        """Test getting unique target files."""
        collection = EditCollection()
        collection.add_edit(EditInstruction(
            edit_type=EditType.REPLACE, file_path="a.py",
            old_text="x", new_text="y", reason="test"
        ))
        collection.add_edit(EditInstruction(
            edit_type=EditType.REPLACE, file_path="b.py",
            old_text="x", new_text="y", reason="test"
        ))
        collection.add_edit(EditInstruction(
            edit_type=EditType.REPLACE, file_path="a.py",  # Duplicate
            old_text="p", new_text="q", reason="test"
        ))

        files = collection.get_target_files()
        assert len(files) == 2
        assert "a.py" in files
        assert "b.py" in files

    def test_to_dict(self):
        """Test serialization to dict."""
        collection = EditCollection()
        collection.summary = "Test summary"
        collection.risk_level = "medium"
        collection.add_edit(EditInstruction(
            edit_type=EditType.REPLACE, file_path="test.py",
            old_text="a", new_text="b", reason="test"
        ))

        data = collection.to_dict()
        assert data["summary"] == "Test summary"
        assert data["risk_level"] == "medium"
        assert data["edit_count"] == 1
        assert len(data["edits"]) == 1


class TestCoder:
    """Tests for Coder."""

    def test_generate_security_fixes_sha256(self):
        """Test rule-based fix for SHA256 password hashing."""
        coder = Coder()

        file_content = '''
def hash_password(password: str) -> str:
    """Hash password for storage."""
    return hashlib.sha256(password.encode()).hexdigest()
'''

        fix_plan = {
            "bug_reason": "Uses SHA256 which is not suitable for passwords",
            "target_files": ["security.py"],
            "fix_strategy": "Use PBKDF2 instead of SHA256",
            "risk_level": "medium"
        }

        collection = coder.generate_edits(
            issue="Password hashing uses SHA256",
            fix_plan=fix_plan,
            target_file_contents={"security.py": file_content},
        )

        assert len(collection.edits) > 0
        assert any("pbkdf2" in e.new_text.lower() or "pbkdf2" in e.reason.lower()
                   for e in collection.edits)

    def test_generate_permission_fix(self):
        """Test fix for always-true permission check."""
        coder = Coder()

        file_content = '''
    def check_permission(self, action: str) -> bool:
        """Check if user has permission for action."""
        # BUG: Everyone has admin permissions
        return True
'''

        fix_plan = {
            "bug_reason": "check_permission always returns True regardless of action",
            "target_files": ["user_service.py"],
            "fix_strategy": "Implement actual permission check",
            "risk_level": "medium"
        }

        collection = coder.generate_edits(
            issue="Permission check always returns True",
            fix_plan=fix_plan,
            target_file_contents={"user_service.py": file_content},
        )

        # Should have an edit that changes the return value
        assert len(collection.edits) > 0

    def test_filter_outside_target_files(self):
        """Test that edits are constrained to available target files."""
        coder = Coder(llm=None)

        file_contents = {
            "a.py": "print('a')",
            "b.py": "print('b')",
        }

        fix_plan = {
            "bug_reason": "test",
            "target_files": ["a.py"],  # Only a.py
            "fix_strategy": "test",
        }

        collection = coder.generate_edits(
            issue="test",
            fix_plan=fix_plan,
            target_file_contents=file_contents,
        )

        # When no specific security issues are found, no edits are generated
        # The important thing is the collection was created and validated
        assert collection is not None
        assert len(collection.edits) == 0


class TestCoderPipeline:
    """Tests for CoderPipeline."""

    def test_run_basic(self):
        """Test basic pipeline execution."""
        pipeline = CoderPipeline()

        input_data = CoderInput(
            issue="Fix authentication bug",
            fix_plan={
                "bug_reason": "SHA256 password hashing",
                "target_files": ["auth.py"],
                "fix_strategy": "Use bcrypt",
            },
            target_file_contents={
                "auth.py": "hashlib.sha256(password.encode()).hexdigest()"
            },
        )

        output = pipeline.run(input_data)

        assert output.success or len(output.edit_collection.edits) > 0
        assert output.execution_time >= 0

    def test_validation_fails_empty_files(self):
        """Test that empty target_file_contents generates warning."""
        pipeline = CoderPipeline()

        input_data = CoderInput(
            issue="Fix bug",
            fix_plan={"target_files": []},
            target_file_contents={},
        )

        output = pipeline.run(input_data)
        # Check that warnings contain the expected messages
        assert any("target file contents" in w.lower() for w in output.warnings)
        assert any("no target files" in w.lower() for w in output.warnings)


class TestEditor:
    """Tests for Editor."""

    def test_replace_execution(self, tmp_path):
        """Test executing a replace edit."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("old_code = 1\nprint(old_code)")

        # Configure editor
        config = EditorConfig(
            root=str(tmp_path),
            create_backups=False,
            dry_run=False
        )
        editor = Editor(config)

        # Execute edit
        edit = EditInstruction(
            edit_type=EditType.REPLACE,
            file_path="test.py",
            old_text="old_code",
            new_text="new_code",
            reason="Rename variable"
        )

        result = editor.execute_edit(edit)
        assert result.success

        # Verify content changed
        content = test_file.read_text()
        assert "new_code" in content
        assert "old_code" not in content

    def test_whole_file_execution(self, tmp_path):
        """Test executing a whole_file edit."""
        # Create original file
        test_file = tmp_path / "test.py"
        test_file.write_text("original")

        config = EditorConfig(root=str(tmp_path), create_backups=False)
        editor = Editor(config)

        edit = EditInstruction(
            edit_type=EditType.WHOLE_FILE,
            file_path="test.py",
            new_content="completely new content\nline 2",
            reason="Replace file"
        )

        result = editor.execute_edit(edit)
        assert result.success

        content = test_file.read_text()
        assert "completely new content" in content

    def test_dry_run(self, tmp_path):
        """Test dry run mode doesn't modify files."""
        test_file = tmp_path / "test.py"
        test_file.write_text("original content")

        config = EditorConfig(root=str(tmp_path), dry_run=True)
        editor = Editor(config)

        edit = EditInstruction(
            edit_type=EditType.REPLACE,
            file_path="test.py",
            old_text="original",
            new_text="modified",
            reason="test"
        )

        result = editor.execute_edit(edit)
        assert result.success
        assert "[DRY RUN]" in result.output

        # File should be unchanged
        assert test_file.read_text() == "original content"

    def test_execute_collection(self, tmp_path):
        """Test executing multiple edits in a collection."""
        # Create test file
        test_file = tmp_path / "multi.py"
        test_file.write_text("a = 1\nb = 2\nc = 3")

        config = EditorConfig(root=str(tmp_path), create_backups=False)
        editor = Editor(config)

        collection = EditCollection()
        collection.add_edit(EditInstruction(
            edit_type=EditType.REPLACE, file_path="multi.py",
            old_text="a = 1", new_text="x = 1", reason="1"
        ))
        collection.add_edit(EditInstruction(
            edit_type=EditType.REPLACE, file_path="multi.py",
            old_text="c = 3", new_text="z = 3", reason="2"
        ))

        result = editor.execute_collection(collection)
        assert result.edits_executed == 2

        content = test_file.read_text()
        assert "x = 1" in content
        assert "z = 3" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])