import streamlit as st
from datetime import datetime

class StateManager:
    @staticmethod
    def initialize_state():
        """Initialize all session state variables"""
        if 'audio_files' not in st.session_state:
            st.session_state.audio_files = {}
        if 'batches' not in st.session_state:
            st.session_state.batches = {}
        if 'current_batch' not in st.session_state:
            st.session_state.current_batch = None
        if 'files_uploaded' not in st.session_state:
            st.session_state.files_uploaded = False
        if 'batch_processed' not in st.session_state:
            st.session_state.batch_processed = False

    @staticmethod
    def create_new_batch(files):
        """Create a new batch with the given files"""
        batch_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        st.session_state.batches[batch_timestamp] = {
            'files': files,
            'processed': False
        }
        st.session_state.current_batch = batch_timestamp
        st.session_state.files_uploaded = True

    @staticmethod
    def mark_batch_processed():
        """Mark the current batch as processed"""
        if st.session_state.current_batch in st.session_state.batches:
            updated_batches = st.session_state.batches.copy()
            updated_batches[st.session_state.current_batch]['processed'] = True
            st.session_state.batches = updated_batches
            st.session_state.batch_processed = True

    @staticmethod
    def get_current_batch_files():
        """Get files from the current batch"""
        if st.session_state.current_batch and st.session_state.current_batch in st.session_state.batches:
            return st.session_state.batches[st.session_state.current_batch]['files']
        return []

    @staticmethod
    def is_batch_processed():
        """Check if current batch is processed"""
        return st.session_state.batch_processed

    @staticmethod
    def get_debug_info():
        """Get debug information about the current state"""
        return {
            'files_uploaded': st.session_state.files_uploaded,
            'current_batch': st.session_state.current_batch,
            'batch_exists': st.session_state.current_batch in st.session_state.batches,
            'batch_processed': st.session_state.batch_processed,
            'batch_keys': list(st.session_state.batches.keys())
        } 