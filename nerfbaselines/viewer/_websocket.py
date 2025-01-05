import logging
import threading
import struct
import hashlib
import os
import base64
import zlib
from enum import Enum, IntEnum
from collections import deque
from codecs import getincrementaldecoder
from typing import Optional, Union, Generator, Tuple, Deque, NamedTuple, Generic, List, TypeVar, cast, Callable
from dataclasses import dataclass
import selectors
from time import time


# The MIT License (MIT)
# 
# Copyright (c) 2017 Benno Rice and contributors
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.



class ConnectionType(Enum):
    """An enumeration of connection types."""

    #: This connection will act as client and talk to a remote server
    CLIENT = 1

    #: This connection will as as server and waits for client connections
    SERVER = 2


class ConnectionState(Enum):
    #: The opening handshake is in progress.
    CONNECTING = 0
    #: The opening handshake is complete.
    OPEN = 1
    #: The remote WebSocket has initiated a connection close.
    REMOTE_CLOSING = 2
    #: The local WebSocket (i.e. this instance) has initiated a connection close.
    LOCAL_CLOSING = 3
    #: The closing handshake has completed.
    CLOSED = 4
    #: The connection was rejected during the opening handshake.
    REJECTING = 5


class CloseReason(IntEnum):
    """
    RFC 6455, Section 7.4.1 - Defined Status Codes
    """

    #: indicates a normal closure, meaning that the purpose for
    #: which the connection was established has been fulfilled.
    NORMAL_CLOSURE = 1000

    #: indicates that an endpoint is "going away", such as a server
    #: going down or a browser having navigated away from a page.
    GOING_AWAY = 1001

    #: indicates that an endpoint is terminating the connection due
    #: to a protocol error.
    PROTOCOL_ERROR = 1002

    #: indicates that an endpoint is terminating the connection
    #: because it has received a type of data it cannot accept (e.g., an
    #: endpoint that understands only text data MAY send this if it
    #: receives a binary message).
    UNSUPPORTED_DATA = 1003

    #: Reserved.  The specific meaning might be defined in the future.
    # DON'T DEFINE THIS: RESERVED_1004 = 1004

    #: is a reserved value and MUST NOT be set as a status code in a
    #: Close control frame by an endpoint.  It is designated for use in
    #: applications expecting a status code to indicate that no status
    #: code was actually present.
    NO_STATUS_RCVD = 1005

    #: is a reserved value and MUST NOT be set as a status code in a
    #: Close control frame by an endpoint.  It is designated for use in
    #: applications expecting a status code to indicate that the
    #: connection was closed abnormally, e.g., without sending or
    #: receiving a Close control frame.
    ABNORMAL_CLOSURE = 1006

    #: indicates that an endpoint is terminating the connection
    #: because it has received data within a message that was not
    #: consistent with the type of the message (e.g., non-UTF-8 [RFC3629]
    #: data within a text message).
    INVALID_FRAME_PAYLOAD_DATA = 1007

    #: indicates that an endpoint is terminating the connection
    #: because it has received a message that violates its policy.  This
    #: is a generic status code that can be returned when there is no
    #: other more suitable status code (e.g., 1003 or 1009) or if there
    #: is a need to hide specific details about the policy.
    POLICY_VIOLATION = 1008

    #: indicates that an endpoint is terminating the connection
    #: because it has received a message that is too big for it to
    #: process.
    MESSAGE_TOO_BIG = 1009

    #: indicates that an endpoint (client) is terminating the
    #: connection because it has expected the server to negotiate one or
    #: more extension, but the server didn't return them in the response
    #: message of the WebSocket handshake.  The list of extensions that
    #: are needed SHOULD appear in the /reason/ part of the Close frame.
    #: Note that this status code is not used by the server, because it
    #: can fail the WebSocket handshake instead.
    MANDATORY_EXT = 1010

    #: indicates that a server is terminating the connection because
    #: it encountered an unexpected condition that prevented it from
    #: fulfilling the request.
    INTERNAL_ERROR = 1011

    #: Server/service is restarting
    #: (not part of RFC6455)
    SERVICE_RESTART = 1012

    #: Temporary server condition forced blocking client's request
    #: (not part of RFC6455)
    TRY_AGAIN_LATER = 1013

    #: is a reserved value and MUST NOT be set as a status code in a
    #: Close control frame by an endpoint.  It is designated for use in
    #: applications expecting a status code to indicate that the
    #: connection was closed due to a failure to perform a TLS handshake
    #: (e.g., the server certificate can't be verified).
    TLS_HANDSHAKE_FAILED = 1015


class Opcode(IntEnum):
    """
    RFC 6455, Section 5.2 - Base Framing Protocol
    """

    #: Continuation frame
    CONTINUATION = 0x0

    #: Text message
    TEXT = 0x1

    #: Binary message
    BINARY = 0x2

    #: Close frame
    CLOSE = 0x8

    #: Ping frame
    PING = 0x9

    #: Pong frame
    PONG = 0xA

    def iscontrol(self) -> bool:
        return bool(self & 0x08)


class Event:
    """
    Base class for wsproto events.
    """

    pass  # noqa


@dataclass(frozen=True)
class CloseConnection(Event):
    """The end of a Websocket connection, represents a closure frame.

    **wsproto does not automatically send a response to a close event.** To
    comply with the RFC you MUST send a close event back to the remote WebSocket
    if you have not already sent one. The :meth:`response` method provides a
    suitable event for this purpose, and you should check if a response needs
    to be sent by checking :func:`wsproto.WSConnection.state`.

    Fields:

    .. attribute:: code

       (Required) The integer close code to indicate why the connection
       has closed.

    .. attribute:: reason

       Additional reasoning for why the connection has closed.

    """

    code: int
    reason: Optional[str] = None

    def response(self) -> "CloseConnection":
        """Generate an RFC-compliant close frame to send back to the peer."""
        return CloseConnection(code=self.code, reason=self.reason)


T = TypeVar("T", bytes, str)


@dataclass(frozen=True)
class Message(Event, Generic[T]):
    """The websocket data message.

    Fields:

    .. attribute:: data

       (Required) The message data as byte string, can be decoded as UTF-8 for
       TEXT messages.  This only represents a single chunk of data and
       not a full WebSocket message.  You need to buffer and
       reassemble these chunks to get the full message.

    .. attribute:: frame_finished

       This has no semantic content, but is provided just in case some
       weird edge case user wants to be able to reconstruct the
       fragmentation pattern of the original stream.

    .. attribute:: message_finished

       True if this frame is the last one of this message, False if
       more frames are expected.

    """

    data: T
    frame_finished: bool = True
    message_finished: bool = True


@dataclass(frozen=True)
class TextMessage(Message[str]):  # pylint: disable=unsubscriptable-object
    """This event is fired when a data frame with TEXT payload is received.

    Fields:

    .. attribute:: data

       The message data as string, This only represents a single chunk
       of data and not a full WebSocket message.  You need to buffer
       and reassemble these chunks to get the full message.

    """

    # https://github.com/python/mypy/issues/5744
    data: str


@dataclass(frozen=True)
class BytesMessage(Message[bytes]):  # pylint: disable=unsubscriptable-object
    """This event is fired when a data frame with BINARY payload is
    received.

    Fields:

    .. attribute:: data

       The message data as byte string, can be decoded as UTF-8 for
       TEXT messages.  This only represents a single chunk of data and
       not a full WebSocket message.  You need to buffer and
       reassemble these chunks to get the full message.
    """

    # https://github.com/python/mypy/issues/5744
    data: bytes


@dataclass(frozen=True)
class Ping(Event):
    """The Ping event can be sent to trigger a ping frame and is fired
    when a Ping is received.

    **wsproto does not automatically send a pong response to a ping event.** To
    comply with the RFC you MUST send a pong even as soon as is practical. The
    :meth:`response` method provides a suitable event for this purpose.

    Fields:

    .. attribute:: payload

       An optional payload to emit with the ping frame.
    """

    payload: bytes = b""

    def response(self) -> "Pong":
        """Generate an RFC-compliant :class:`Pong` response to this ping."""
        return Pong(payload=self.payload)


@dataclass(frozen=True)
class Pong(Event):
    """The Pong event is fired when a Pong is received.

    Fields:

    .. attribute:: payload

       An optional payload to emit with the pong frame.

    """

    payload: bytes = b""


class ProtocolError(Exception):
    pass


class LocalProtocolError(ProtocolError):
    """Indicates an error due to local/programming errors.

    This is raised when the connection is asked to do something that
    is either incompatible with the state or the websocket standard.

    """

    pass  # noqa


class RemoteProtocolError(ProtocolError):
    """Indicates an error due to the remote's actions.

    This is raised when processing the bytes from the remote if the
    remote has sent data that is incompatible with the websocket
    standard.

    .. attribute:: event_hint

       This is a suggested wsproto Event to send to the client based
       on the error. It could be None if no hint is available.

    """

    def __init__(self, message: str, event_hint: Optional[Event] = None) -> None:
        self.event_hint = event_hint
        super().__init__(message)


class ParseFailed(Exception):
    def __init__(
        self, msg: str, code: CloseReason = CloseReason.PROTOCOL_ERROR
    ) -> None:
        super().__init__(msg)
        self.code = code


# RFC 6455, Section 7.4.1 - Defined Status Codes
LOCAL_ONLY_CLOSE_REASONS = (
    CloseReason.NO_STATUS_RCVD,
    CloseReason.ABNORMAL_CLOSURE,
    CloseReason.TLS_HANDSHAKE_FAILED,
)

# RFC 6455, Section 7.4.2 - Status Code Ranges
MIN_CLOSE_REASON = 1000
MIN_PROTOCOL_CLOSE_REASON = 1000
MAX_PROTOCOL_CLOSE_REASON = 2999
MIN_LIBRARY_CLOSE_REASON = 3000
MAX_LIBRARY_CLOSE_REASON = 3999
MIN_PRIVATE_CLOSE_REASON = 4000
MAX_PRIVATE_CLOSE_REASON = 4999
MAX_CLOSE_REASON = 4999

# RFC6455, Section 5.2 - Base Framing Protocol

# Payload length constants
PAYLOAD_LENGTH_TWO_BYTE = 126
PAYLOAD_LENGTH_EIGHT_BYTE = 127
MAX_PAYLOAD_NORMAL = 125
MAX_PAYLOAD_TWO_BYTE = 2**16 - 1
MAX_PAYLOAD_EIGHT_BYTE = 2**64 - 1
MAX_FRAME_PAYLOAD = MAX_PAYLOAD_EIGHT_BYTE

# MASK and PAYLOAD LEN are packed into a byte
MASK_MASK = 0x80
PAYLOAD_LEN_MASK = 0x7F

# FIN, RSV[123] and OPCODE are packed into a single byte
FIN_MASK = 0x80
RSV1_MASK = 0x40
RSV2_MASK = 0x20
RSV3_MASK = 0x10
OPCODE_MASK = 0x0F


NULL_MASK = struct.pack("!I", 0)


class RsvBits(NamedTuple):
    rsv1: bool
    rsv2: bool
    rsv3: bool


class Header(NamedTuple):
    fin: bool
    rsv: RsvBits
    opcode: Opcode
    payload_len: int
    masking_key: Optional[bytes]


class Frame(NamedTuple):
    opcode: Opcode
    payload: Union[bytes, str, Tuple[int, str]]
    frame_finished: bool
    message_finished: bool


def _truncate_utf8(data: bytes, nbytes: int) -> bytes:
    if len(data) <= nbytes:
        return data

    # Truncate
    data = data[:nbytes]
    # But we might have cut a codepoint in half, in which case we want to
    # discard the partial character so the data is at least
    # well-formed. This is a little inefficient since it processes the
    # whole message twice when in theory we could just peek at the last
    # few characters, but since this is only used for close messages (max
    # length = 125 bytes) it really doesn't matter.
    data = data.decode("utf-8", errors="ignore").encode("utf-8")
    return data


class Buffer:
    def __init__(self, initial_bytes: Optional[bytes] = None) -> None:
        self.buffer = bytearray()
        self.bytes_used = 0
        if initial_bytes:
            self.feed(initial_bytes)

    def feed(self, new_bytes: bytes) -> None:
        self.buffer += new_bytes

    def consume_at_most(self, nbytes: int) -> bytes:
        if not nbytes:
            return bytearray()

        data = self.buffer[self.bytes_used : self.bytes_used + nbytes]
        self.bytes_used += len(data)
        return data

    def consume_exactly(self, nbytes: int) -> Optional[bytes]:
        if len(self.buffer) - self.bytes_used < nbytes:
            return None

        return self.consume_at_most(nbytes)

    def commit(self) -> None:
        # In CPython 3.4+, del[:n] is amortized O(n), *not* quadratic
        del self.buffer[: self.bytes_used]
        self.bytes_used = 0

    def rollback(self) -> None:
        self.bytes_used = 0

    def __len__(self) -> int:
        return len(self.buffer)


class MessageDecoder:
    def __init__(self) -> None:
        self.opcode = None
        self.decoder = None

    def process_frame(self, frame: Frame) -> Frame:
        assert not frame.opcode.iscontrol()

        if self.opcode is None:
            if frame.opcode is Opcode.CONTINUATION:
                raise ParseFailed("unexpected CONTINUATION")
            self.opcode = frame.opcode
        elif frame.opcode is not Opcode.CONTINUATION:
            raise ParseFailed("expected CONTINUATION, got %r" % frame.opcode)

        if frame.opcode is Opcode.TEXT:
            self.decoder = getincrementaldecoder("utf-8")()

        finished = frame.frame_finished and frame.message_finished

        if self.decoder is None:
            data = frame.payload
        else:
            assert isinstance(frame.payload, (bytes, bytearray))
            try:
                data = self.decoder.decode(frame.payload, finished)
            except UnicodeDecodeError as exc:
                raise ParseFailed(str(exc), CloseReason.INVALID_FRAME_PAYLOAD_DATA)

        frame = Frame(self.opcode, data, frame.frame_finished, finished)

        if finished:
            self.opcode = None
            self.decoder = None

        return frame


_XOR_TABLE = [bytes(a ^ b for a in range(256)) for b in range(256)]


class XorMaskerSimple:
    def __init__(self, masking_key: bytes) -> None:
        self._masking_key = masking_key

    def process(self, data: bytes) -> bytes:
        if data:
            data_array = bytearray(data)
            a, b, c, d = (_XOR_TABLE[n] for n in self._masking_key)
            data_array[::4] = data_array[::4].translate(a)
            data_array[1::4] = data_array[1::4].translate(b)
            data_array[2::4] = data_array[2::4].translate(c)
            data_array[3::4] = data_array[3::4].translate(d)

            # Rotate the masking key so that the next usage continues
            # with the next key element, rather than restarting.
            key_rotation = len(data) % 4
            self._masking_key = (
                self._masking_key[key_rotation:] + self._masking_key[:key_rotation]
            )

            return bytes(data_array)
        return data


class XorMaskerNull:
    def process(self, data: bytes) -> bytes:
        return data


class FrameDecoder:
    def __init__(self, client: bool, extensions: Optional[List] = None) -> None:
        self.client = client
        self.extensions = extensions or []

        self.buffer = Buffer()

        self.header: Optional[Header] = None
        self.effective_opcode: Optional[Opcode] = None
        self.masker: Union[None, XorMaskerNull, XorMaskerSimple] = None
        self.payload_required = 0
        self.payload_consumed = 0

    def receive_bytes(self, data: bytes) -> None:
        self.buffer.feed(data)

    def process_buffer(self) -> Optional[Frame]:
        if not self.header:
            if not self.parse_header():
                return None
        # parse_header() sets these.
        assert self.header is not None
        assert self.masker is not None
        assert self.effective_opcode is not None

        if len(self.buffer) < self.payload_required:
            return None

        payload_remaining = self.header.payload_len - self.payload_consumed
        payload = self.buffer.consume_at_most(payload_remaining)
        if not payload and self.header.payload_len > 0:
            return None
        self.buffer.commit()

        self.payload_consumed += len(payload)
        finished = self.payload_consumed == self.header.payload_len

        payload = self.masker.process(payload)

        for extension in self.extensions:
            payload_ = extension.frame_inbound_payload_data(self, payload)
            if isinstance(payload_, CloseReason):
                raise ParseFailed("error in extension", payload_)
            payload = payload_

        if finished:
            final = bytearray()
            for extension in self.extensions:
                result = extension.frame_inbound_complete(self, self.header.fin)
                if isinstance(result, CloseReason):
                    raise ParseFailed("error in extension", result)
                if result is not None:
                    final += result
            payload += final

        frame = Frame(self.effective_opcode, payload, finished, self.header.fin)

        if finished:
            self.header = None
            self.effective_opcode = None
            self.masker = None
        else:
            self.effective_opcode = Opcode.CONTINUATION

        return frame

    def parse_header(self) -> bool:
        data = self.buffer.consume_exactly(2)
        if data is None:
            self.buffer.rollback()
            return False

        fin = bool(data[0] & FIN_MASK)
        rsv = RsvBits(
            bool(data[0] & RSV1_MASK),
            bool(data[0] & RSV2_MASK),
            bool(data[0] & RSV3_MASK),
        )
        opcode = data[0] & OPCODE_MASK
        try:
            opcode = Opcode(opcode)
        except ValueError:
            raise ParseFailed(f"Invalid opcode {opcode:#x}")

        if opcode.iscontrol() and not fin:
            raise ParseFailed("Invalid attempt to fragment control frame")

        has_mask = bool(data[1] & MASK_MASK)
        payload_len_short = data[1] & PAYLOAD_LEN_MASK
        payload_len = self.parse_extended_payload_length(opcode, payload_len_short)
        if payload_len is None:
            self.buffer.rollback()
            return False

        self.extension_processing(opcode, rsv, payload_len)

        if has_mask and self.client:
            raise ParseFailed("client received unexpected masked frame")
        if not has_mask and not self.client:
            raise ParseFailed("server received unexpected unmasked frame")
        if has_mask:
            masking_key = self.buffer.consume_exactly(4)
            if masking_key is None:
                self.buffer.rollback()
                return False
            self.masker = XorMaskerSimple(masking_key)
        else:
            self.masker = XorMaskerNull()

        self.buffer.commit()
        self.header = Header(fin, rsv, opcode, payload_len, None)
        self.effective_opcode = self.header.opcode
        if self.header.opcode.iscontrol():
            self.payload_required = payload_len
        else:
            self.payload_required = 0
        self.payload_consumed = 0
        return True

    def parse_extended_payload_length(
        self, opcode: Opcode, payload_len: int
    ) -> Optional[int]:
        if opcode.iscontrol() and payload_len > MAX_PAYLOAD_NORMAL:
            raise ParseFailed("Control frame with payload len > 125")
        if payload_len == PAYLOAD_LENGTH_TWO_BYTE:
            data = self.buffer.consume_exactly(2)
            if data is None:
                return None
            (payload_len,) = struct.unpack("!H", data)
            if payload_len <= MAX_PAYLOAD_NORMAL:
                raise ParseFailed(
                    "Payload length used 2 bytes when 1 would have sufficed"
                )
        elif payload_len == PAYLOAD_LENGTH_EIGHT_BYTE:
            data = self.buffer.consume_exactly(8)
            if data is None:
                return None
            (payload_len,) = struct.unpack("!Q", data)
            if payload_len <= MAX_PAYLOAD_TWO_BYTE:
                raise ParseFailed(
                    "Payload length used 8 bytes when 2 would have sufficed"
                )
            if payload_len >> 63:
                # I'm not sure why this is illegal, but that's what the RFC
                # says, so...
                raise ParseFailed("8-byte payload length with non-zero MSB")

        return payload_len

    def extension_processing(
        self, opcode: Opcode, rsv: RsvBits, payload_len: int
    ) -> None:
        rsv_used = [False, False, False]
        for extension in self.extensions:
            result = extension.frame_inbound_header(self, opcode, rsv, payload_len)
            if isinstance(result, CloseReason):
                raise ParseFailed("error in extension", result)
            for bit, used in enumerate(result):
                if used:
                    rsv_used[bit] = True
        for expected, found in zip(rsv_used, rsv):
            if found and not expected:
                raise ParseFailed("Reserved bit set unexpectedly")



class FrameProtocol:
    def __init__(self, client: bool, extensions: List) -> None:
        self.client = client
        self.extensions = [ext for ext in extensions if ext.enabled()]

        # Global state
        self._frame_decoder = FrameDecoder(self.client, self.extensions)
        self._message_decoder = MessageDecoder()
        self._parse_more = self._parse_more_gen()

        self._outbound_opcode: Optional[Opcode] = None

    def _process_close(self, frame: Frame) -> Frame:
        data = frame.payload
        assert isinstance(data, (bytes, bytearray))

        if not data:
            # "If this Close control frame contains no status code, _The
            # WebSocket Connection Close Code_ is considered to be 1005"
            data = (CloseReason.NO_STATUS_RCVD, "")
        elif len(data) == 1:
            raise ParseFailed("CLOSE with 1 byte payload")
        else:
            (code,) = struct.unpack("!H", data[:2])
            if code < MIN_CLOSE_REASON or code > MAX_CLOSE_REASON:
                raise ParseFailed("CLOSE with invalid code")
            try:
                code = CloseReason(code)
            except ValueError:
                pass
            if code in LOCAL_ONLY_CLOSE_REASONS:
                raise ParseFailed("remote CLOSE with local-only reason")
            if not isinstance(code, CloseReason) and code <= MAX_PROTOCOL_CLOSE_REASON:
                raise ParseFailed("CLOSE with unknown reserved code")
            try:
                reason = data[2:].decode("utf-8")
            except UnicodeDecodeError as exc:
                raise ParseFailed(
                    "Error decoding CLOSE reason: " + str(exc),
                    CloseReason.INVALID_FRAME_PAYLOAD_DATA,
                )
            data = (code, reason)

        return Frame(frame.opcode, data, frame.frame_finished, frame.message_finished)

    def _parse_more_gen(self) -> Generator[Optional[Frame], None, None]:
        # Consume as much as we can from self._buffer, yielding events, and
        # then yield None when we need more data. Or raise ParseFailed.

        # XX FIXME this should probably be refactored so that we never see
        # disabled extensions in the first place...
        self.extensions = [ext for ext in self.extensions if ext.enabled()]
        closed = False

        while not closed:
            frame = self._frame_decoder.process_buffer()

            if frame is not None:
                if not frame.opcode.iscontrol():
                    frame = self._message_decoder.process_frame(frame)
                elif frame.opcode == Opcode.CLOSE:
                    frame = self._process_close(frame)
                    closed = True

            yield frame

    def receive_bytes(self, data: bytes) -> None:
        self._frame_decoder.receive_bytes(data)

    def received_frames(self) -> Generator[Frame, None, None]:
        for event in self._parse_more:
            if event is None:
                break
            else:
                yield event

    def close(self, code: Optional[int] = None, reason: Optional[str] = None) -> bytes:
        payload = bytearray()
        if code is CloseReason.NO_STATUS_RCVD:
            code = None
        if code is None and reason:
            raise TypeError("cannot specify a reason without a code")
        if code in LOCAL_ONLY_CLOSE_REASONS:
            code = CloseReason.NORMAL_CLOSURE
        if code is not None:
            payload += bytearray(struct.pack("!H", code))
            if reason is not None:
                payload += _truncate_utf8(
                    reason.encode("utf-8"), MAX_PAYLOAD_NORMAL - 2
                )

        return self._serialize_frame(Opcode.CLOSE, payload)

    def ping(self, payload: bytes = b"") -> bytes:
        return self._serialize_frame(Opcode.PING, payload)

    def pong(self, payload: bytes = b"") -> bytes:
        return self._serialize_frame(Opcode.PONG, payload)

    def send_data(
        self, payload: Union[bytes, bytearray, str] = b"", fin: bool = True
    ) -> bytes:
        if isinstance(payload, (bytes, bytearray, memoryview)):
            opcode = Opcode.BINARY
        elif isinstance(payload, str):
            opcode = Opcode.TEXT
            payload = payload.encode("utf-8")
        else:
            raise ValueError("Must provide bytes or text")

        if self._outbound_opcode is None:
            self._outbound_opcode = opcode
        elif self._outbound_opcode is not opcode:
            raise TypeError("Data type mismatch inside message")
        else:
            opcode = Opcode.CONTINUATION

        if fin:
            self._outbound_opcode = None

        return self._serialize_frame(opcode, payload, fin)

    def _make_fin_rsv_opcode(self, fin: bool, rsv: RsvBits, opcode: Opcode) -> int:
        fin_bits = int(fin) << 7
        rsv_bits = (int(rsv.rsv1) << 6) + (int(rsv.rsv2) << 5) + (int(rsv.rsv3) << 4)
        opcode_bits = int(opcode)

        return fin_bits | rsv_bits | opcode_bits

    def _serialize_frame(
        self, opcode: Opcode, payload: bytes = b"", fin: bool = True
    ) -> bytes:
        rsv = RsvBits(False, False, False)
        for extension in reversed(self.extensions):
            rsv, payload = extension.frame_outbound(self, opcode, rsv, payload, fin)

        fin_rsv_opcode = self._make_fin_rsv_opcode(fin, rsv, opcode)

        payload_length = len(payload)
        quad_payload = False
        if payload_length <= MAX_PAYLOAD_NORMAL:
            first_payload = payload_length
            second_payload = None
        elif payload_length <= MAX_PAYLOAD_TWO_BYTE:
            first_payload = PAYLOAD_LENGTH_TWO_BYTE
            second_payload = payload_length
        else:
            first_payload = PAYLOAD_LENGTH_EIGHT_BYTE
            second_payload = payload_length
            quad_payload = True

        if self.client:
            first_payload |= 1 << 7

        header = bytearray([fin_rsv_opcode, first_payload])
        if second_payload is not None:
            if opcode.iscontrol():
                raise ValueError("payload too long for control frame")
            if quad_payload:
                header += bytearray(struct.pack("!Q", second_payload))
            else:
                header += bytearray(struct.pack("!H", second_payload))

        if self.client:
            # "The masking key is a 32-bit value chosen at random by the
            # client.  When preparing a masked frame, the client MUST pick a
            # fresh masking key from the set of allowed 32-bit values.  The
            # masking key needs to be unpredictable; thus, the masking key
            # MUST be derived from a strong source of entropy, and the masking
            # key for a given frame MUST NOT make it simple for a server/proxy
            # to predict the masking key for a subsequent frame.  The
            # unpredictability of the masking key is essential to prevent
            # authors of malicious applications from selecting the bytes that
            # appear on the wire."
            #   -- https://tools.ietf.org/html/rfc6455#section-5.3
            masking_key = os.urandom(4)
            masker = XorMaskerSimple(masking_key)
            return header + masking_key + masker.process(payload)

        return header + payload


class Connection:
    def __init__(
        self,
        connection_type: ConnectionType,
        extensions = None,
        trailing_data: bytes = b"",
    ) -> None:
        self.client = connection_type is ConnectionType.CLIENT
        self._events: Deque[Event] = deque()
        self._proto = FrameProtocol(self.client, extensions or [])
        self._state = ConnectionState.OPEN
        self.receive_data(trailing_data)

    @property
    def state(self) -> ConnectionState:
        return self._state

    def send(self, event: Event) -> bytes:
        data = b""
        if isinstance(event, Message) and self.state == ConnectionState.OPEN:
            data += self._proto.send_data(event.data, event.message_finished)
        elif isinstance(event, Ping) and self.state == ConnectionState.OPEN:
            data += self._proto.ping(event.payload)
        elif isinstance(event, Pong) and self.state == ConnectionState.OPEN:
            data += self._proto.pong(event.payload)
        elif isinstance(event, CloseConnection) and self.state in {
            ConnectionState.OPEN,
            ConnectionState.REMOTE_CLOSING,
        }:
            data += self._proto.close(event.code, event.reason)
            if self.state == ConnectionState.REMOTE_CLOSING:
                self._state = ConnectionState.CLOSED
            else:
                self._state = ConnectionState.LOCAL_CLOSING
        else:
            raise LocalProtocolError(
                f"Event {event} cannot be sent in state {self.state}."
            )
        return data

    def receive_data(self, data: Optional[bytes]) -> None:
        """
        Pass some received data to the connection for handling.

        A list of events that the remote peer triggered by sending this data can
        be retrieved with :meth:`~wsproto.connection.Connection.events`.

        :param data: The data received from the remote peer on the network.
        :type data: ``bytes``
        """

        if data is None:
            # "If _The WebSocket Connection is Closed_ and no Close control
            # frame was received by the endpoint (such as could occur if the
            # underlying transport connection is lost), _The WebSocket
            # Connection Close Code_ is considered to be 1006."
            self._events.append(CloseConnection(code=CloseReason.ABNORMAL_CLOSURE))
            self._state = ConnectionState.CLOSED
            return

        if self.state in (ConnectionState.OPEN, ConnectionState.LOCAL_CLOSING):
            self._proto.receive_bytes(data)
        elif self.state is ConnectionState.CLOSED:
            raise LocalProtocolError("Connection already closed.")
        else:
            pass  # pragma: no cover

    def events(self) -> Generator[Event, None, None]:
        """
        Return a generator that provides any events that have been generated
        by protocol activity.

        :returns: generator of :class:`Event <wsproto.events.Event>` subclasses
        """
        while self._events:
            yield self._events.popleft()

        try:
            for frame in self._proto.received_frames():
                if frame.opcode is Opcode.PING:
                    assert frame.frame_finished and frame.message_finished
                    assert isinstance(frame.payload, (bytes, bytearray))
                    yield Ping(payload=frame.payload)

                elif frame.opcode is Opcode.PONG:
                    assert frame.frame_finished and frame.message_finished
                    assert isinstance(frame.payload, (bytes, bytearray))
                    yield Pong(payload=frame.payload)

                elif frame.opcode is Opcode.CLOSE:
                    assert isinstance(frame.payload, tuple)
                    code, reason = frame.payload
                    if self.state is ConnectionState.LOCAL_CLOSING:
                        self._state = ConnectionState.CLOSED
                    else:
                        self._state = ConnectionState.REMOTE_CLOSING
                    yield CloseConnection(code=code, reason=reason)

                elif frame.opcode is Opcode.TEXT:
                    assert isinstance(frame.payload, str)
                    yield TextMessage(
                        data=frame.payload,
                        frame_finished=frame.frame_finished,
                        message_finished=frame.message_finished,
                    )

                elif frame.opcode is Opcode.BINARY:
                    assert isinstance(frame.payload, (bytes, bytearray))
                    yield BytesMessage(
                        data=frame.payload,
                        frame_finished=frame.frame_finished,
                        message_finished=frame.message_finished,
                    )

                else:
                    pass  # pragma: no cover
        except ParseFailed as exc:
            yield CloseConnection(code=exc.code, reason=str(exc))


# RFC6455, Section 1.3 - Opening Handshake
ACCEPT_GUID = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"

# RFC6455, Section 4.2.1/6 - Reading the Client's Opening Handshake
WEBSOCKET_VERSION = b"13"

# RFC6455, Section 4.2.1/3 - Value of the Upgrade header
WEBSOCKET_UPGRADE = b"websocket"


class PerMessageDeflate:
    name = "permessage-deflate"

    DEFAULT_CLIENT_MAX_WINDOW_BITS = 15
    DEFAULT_SERVER_MAX_WINDOW_BITS = 15

    def __init__(
        self,
        client_no_context_takeover: bool = False,
        client_max_window_bits: Optional[int] = None,
        server_no_context_takeover: bool = False,
        server_max_window_bits: Optional[int] = None,
    ) -> None:
        self.client_no_context_takeover = client_no_context_takeover
        self.server_no_context_takeover = server_no_context_takeover
        self._client_max_window_bits = self.DEFAULT_CLIENT_MAX_WINDOW_BITS
        self._server_max_window_bits = self.DEFAULT_SERVER_MAX_WINDOW_BITS
        if client_max_window_bits is not None:
            self.client_max_window_bits = client_max_window_bits
        if server_max_window_bits is not None:
            self.server_max_window_bits = server_max_window_bits

        self._compressor = None  # noqa
        self._decompressor = None  # noqa
        # This refers to the current frame
        self._inbound_is_compressible: Optional[bool] = None
        # This refers to the ongoing message (which might span multiple
        # frames). Only the first frame in a fragmented message is flagged for
        # compression, so this carries that bit forward.
        self._inbound_compressed: Optional[bool] = None

        self._enabled = False

    @property
    def client_max_window_bits(self) -> int:
        return self._client_max_window_bits

    @client_max_window_bits.setter
    def client_max_window_bits(self, value: int) -> None:
        if value < 9 or value > 15:
            raise ValueError("Window size must be between 9 and 15 inclusive")
        self._client_max_window_bits = value

    @property
    def server_max_window_bits(self) -> int:
        return self._server_max_window_bits

    @server_max_window_bits.setter
    def server_max_window_bits(self, value: int) -> None:
        if value < 9 or value > 15:
            raise ValueError("Window size must be between 9 and 15 inclusive")
        self._server_max_window_bits = value

    def _compressible_opcode(self, opcode) -> bool:
        return opcode in (Opcode.TEXT, Opcode.BINARY, Opcode.CONTINUATION)

    def enabled(self) -> bool:
        return self._enabled

    def offer(self) -> Union[bool, str]:
        parameters = [
            "client_max_window_bits=%d" % self.client_max_window_bits,
            "server_max_window_bits=%d" % self.server_max_window_bits,
        ]

        if self.client_no_context_takeover:
            parameters.append("client_no_context_takeover")
        if self.server_no_context_takeover:
            parameters.append("server_no_context_takeover")

        return "; ".join(parameters)

    def finalize(self, offer: str) -> None:
        bits = [b.strip() for b in offer.split(";")]
        for bit in bits[1:]:
            if bit.startswith("client_no_context_takeover"):
                self.client_no_context_takeover = True
            elif bit.startswith("server_no_context_takeover"):
                self.server_no_context_takeover = True
            elif bit.startswith("client_max_window_bits"):
                self.client_max_window_bits = int(bit.split("=", 1)[1].strip())
            elif bit.startswith("server_max_window_bits"):
                self.server_max_window_bits = int(bit.split("=", 1)[1].strip())

        self._enabled = True

    def _parse_params(self, params: str):
        client_max_window_bits = None
        server_max_window_bits = None

        bits = [b.strip() for b in params.split(";")]
        for bit in bits[1:]:
            if bit.startswith("client_no_context_takeover"):
                self.client_no_context_takeover = True
            elif bit.startswith("server_no_context_takeover"):
                self.server_no_context_takeover = True
            elif bit.startswith("client_max_window_bits"):
                if "=" in bit:
                    client_max_window_bits = int(bit.split("=", 1)[1].strip())
                else:
                    client_max_window_bits = self.client_max_window_bits
            elif bit.startswith("server_max_window_bits"):
                if "=" in bit:
                    server_max_window_bits = int(bit.split("=", 1)[1].strip())
                else:
                    server_max_window_bits = self.server_max_window_bits

        return client_max_window_bits, server_max_window_bits

    def accept(self, offer: str) -> Union[bool, None, str]:
        client_max_window_bits, server_max_window_bits = self._parse_params(offer)

        parameters = []

        if self.client_no_context_takeover:
            parameters.append("client_no_context_takeover")
        if self.server_no_context_takeover:
            parameters.append("server_no_context_takeover")
        try:
            if client_max_window_bits is not None:
                parameters.append("client_max_window_bits=%d" % client_max_window_bits)
                self.client_max_window_bits = client_max_window_bits
            if server_max_window_bits is not None:
                parameters.append("server_max_window_bits=%d" % server_max_window_bits)
                self.server_max_window_bits = server_max_window_bits
        except ValueError:
            return None
        else:
            self._enabled = True
            if not parameters:
                return True
            return "; ".join(parameters)

    def frame_inbound_header(self, proto, opcode, rsv, payload_length):
        del payload_length
        if rsv.rsv1 and opcode.iscontrol():
            return CloseReason.PROTOCOL_ERROR
        if rsv.rsv1 and opcode is Opcode.CONTINUATION:
            return CloseReason.PROTOCOL_ERROR

        self._inbound_is_compressible = self._compressible_opcode(opcode)

        if self._inbound_compressed is None:
            self._inbound_compressed = rsv.rsv1
            if self._inbound_compressed:
                assert self._inbound_is_compressible
                if proto.client:
                    bits = self.server_max_window_bits
                else:
                    bits = self.client_max_window_bits
                if self._decompressor is None:
                    self._decompressor = zlib.decompressobj(-int(bits))

        return RsvBits(True, False, False)

    def frame_inbound_payload_data(self, proto, data: bytes) -> Union[bytes, CloseReason]:
        del proto
        if not self._inbound_compressed or not self._inbound_is_compressible:
            return data
        assert self._decompressor is not None

        try:
            return self._decompressor.decompress(bytes(data))
        except zlib.error:
            return CloseReason.INVALID_FRAME_PAYLOAD_DATA

    def frame_inbound_complete(self, proto, fin: bool) -> Union[bytes, CloseReason, None]:
        if not fin:
            return None
        if not self._inbound_is_compressible:
            self._inbound_compressed = None
            return None
        if not self._inbound_compressed:
            self._inbound_compressed = None
            return None
        assert self._decompressor is not None

        try:
            data = self._decompressor.decompress(b"\x00\x00\xff\xff")
            data += self._decompressor.flush()
        except zlib.error:
            return CloseReason.INVALID_FRAME_PAYLOAD_DATA

        if proto.client:
            no_context_takeover = self.server_no_context_takeover
        else:
            no_context_takeover = self.client_no_context_takeover

        if no_context_takeover:
            self._decompressor = None

        self._inbound_compressed = None

        return data

    def frame_outbound(self, proto, opcode: Opcode, rsv: RsvBits, data: bytes, fin: bool) -> Tuple[RsvBits, bytes]:
        if not self._compressible_opcode(opcode):
            return (rsv, data)

        if opcode is not Opcode.CONTINUATION:
            rsv = RsvBits(True, *rsv[1:])

        if self._compressor is None:
            assert opcode is not Opcode.CONTINUATION
            if proto.client:
                bits = self.client_max_window_bits
            else:
                bits = self.server_max_window_bits
            self._compressor = zlib.compressobj(
                zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -int(bits)
            )

        data = self._compressor.compress(bytes(data))

        if fin:
            data += self._compressor.flush(zlib.Z_SYNC_FLUSH)
            data = data[:-4]

            if proto.client:
                no_context_takeover = self.client_no_context_takeover
            else:
                no_context_takeover = self.server_no_context_takeover

            if no_context_takeover:
                self._compressor = None

        return (rsv, data)

    def __repr__(self) -> str:
        descr = ["client_max_window_bits=%d" % self.client_max_window_bits]
        if self.client_no_context_takeover:
            descr.append("client_no_context_takeover")
        descr.append("server_max_window_bits=%d" % self.server_max_window_bits)
        if self.server_no_context_takeover:
            descr.append("server_no_context_takeover")

        return "<{} {}>".format(self.__class__.__name__, "; ".join(descr))


# MIT License
# 
# Copyright (c) 2021 Miguel Grinberg
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.




class SimpleWebsocketError(RuntimeError):
    pass


class ConnectionError(SimpleWebsocketError):
    def __init__(self, status_code=None):  # pragma: no cover
        self.status_code = status_code
        super().__init__(f'Connection error: {status_code}')


class ConnectionClosed(SimpleWebsocketError):
    def __init__(self, reason=CloseReason.NO_STATUS_RCVD, message=None):
        self.reason = reason
        self.message = message
        super().__init__(f'Connection closed: {reason} {message or ""}')


def _get_sock(environ):
    mode = 'unknown'
    sock = None
    if 'werkzeug.socket' in environ:
        # extract socket from Werkzeug's WSGI environment
        sock = environ.get('werkzeug.socket')
        mode = 'werkzeug'
    elif 'gunicorn.socket' in environ:
        # extract socket from Gunicorn WSGI environment
        sock = environ.get('gunicorn.socket')
        mode = 'gunicorn'
    elif 'eventlet.input' in environ:  # pragma: no cover
        # extract socket from Eventlet's WSGI environment
        sock = environ.get('eventlet.input').get_socket()
        mode = 'eventlet'
    elif environ.get('SERVER_SOFTWARE', '').startswith(
            'gevent'):  # pragma: no cover
        # extract socket from Gevent's WSGI environment
        wsgi_input = environ['wsgi.input']
        if not hasattr(wsgi_input, 'raw') and hasattr(wsgi_input, 'rfile'):
            wsgi_input = wsgi_input.rfile
        if hasattr(wsgi_input, 'raw'):
            sock = wsgi_input.raw._sock
            try:
                sock = sock.dup()
            except NotImplementedError:
                pass
            mode = 'gevent'
    if sock is None:
        raise RuntimeError('Cannot obtain socket from WSGI environment.')
    return sock, mode


class Server:
    def __init__(self, sock, receive_bytes=4096, ping_interval=None, max_message_size=None, extensions=None):
        self.sock = sock
        self.receive_bytes = receive_bytes
        self.ping_interval = ping_interval
        self.max_message_size = max_message_size
        self.pong_received = True
        self.input_buffer = []
        self.incoming_message = None
        self.incoming_message_len = 0
        self.connected = True
        self.close_reason = CloseReason.NO_STATUS_RCVD
        self.close_message = None

        self.event = threading.Event()
        self.ws = Connection(ConnectionType.SERVER, extensions=extensions)

        self.thread = threading.Thread(target=self._thread)
        self.thread.start()

    def send(self, data):
        if not self.connected:
            raise ConnectionClosed(cast(CloseReason, self.close_reason), self.close_message)
        if isinstance(data, bytes):
            out_data = self.ws.send(Message(data=data))
        else:
            out_data = self.ws.send(TextMessage(data=str(data)))
        self.sock.send(out_data)

    def receive(self, timeout=None):
        while self.connected and not self.input_buffer:
            if not self.event.wait(timeout=timeout):
                return None
            self.event.clear()
        try:
            return self.input_buffer.pop(0)
        except IndexError:
            pass
        if not self.connected:  # pragma: no cover
            raise ConnectionClosed(cast(CloseReason, self.close_reason), self.close_message)

    def close(self, reason=None, message=None):
        if not self.connected:
            raise ConnectionClosed(cast(CloseReason, self.close_reason), self.close_message)
        out_data = self.ws.send(CloseConnection(
            reason or CloseReason.NORMAL_CLOSURE, message))
        try:
            self.sock.send(out_data)
        except BrokenPipeError:  # pragma: no cover
            pass
        self.connected = False

    def _thread(self):
        sel = None
        next_ping = -float('inf')
        if self.ping_interval:
            next_ping = time() + self.ping_interval
            sel = selectors.DefaultSelector()
            try:
                sel.register(self.sock, selectors.EVENT_READ, True)
            except ValueError:  # pragma: no cover
                self.connected = False

        while self.connected:
            try:
                if sel:
                    now = time()
                    if next_ping <= now or not sel.select(next_ping - now):
                        # we reached the timeout, we have to send a ping
                        if not self.pong_received:
                            self.close(reason=CloseReason.POLICY_VIOLATION,
                                       message='Ping/Pong timeout')
                            self.event.set()
                            break
                        self.pong_received = False
                        self.sock.send(self.ws.send(Ping()))
                        assert self.ping_interval is not None
                        next_ping = max(now, next_ping) + self.ping_interval
                        continue
                in_data = self.sock.recv(self.receive_bytes)
                if len(in_data) == 0:
                    raise OSError()
                self.ws.receive_data(in_data)
                self.connected = self._handle_events()
            except (OSError, ConnectionResetError,
                    LocalProtocolError):  # pragma: no cover
                self.connected = False
                self.event.set()
                break
        sel.close() if sel else None
        self.sock.close()

    def _handle_events(self):
        keep_going = True
        out_data = b''
        for event in self.ws.events():
            try:
                if isinstance(event, CloseConnection):
                    out_data += self.ws.send(event.response())
                    self.close_reason = event.code
                    self.close_message = event.reason
                    self.connected = False
                    self.event.set()
                    keep_going = False
                elif isinstance(event, Ping):
                    out_data += self.ws.send(event.response())
                elif isinstance(event, Pong):
                    self.pong_received = True
                elif isinstance(event, (TextMessage, BytesMessage)):
                    self.incoming_message_len += len(event.data)
                    if self.max_message_size and \
                            self.incoming_message_len > self.max_message_size:
                        out_data += self.ws.send(CloseConnection(
                            CloseReason.MESSAGE_TOO_BIG, 'Message is too big'))
                        self.event.set()
                        keep_going = False
                        break
                    if self.incoming_message is None:
                        # store message as is first
                        # if it is the first of a group, the message will be
                        # converted to bytearray on arrival of the second
                        # part, since bytearrays are mutable and can be
                        # concatenated more efficiently
                        self.incoming_message = event.data
                    elif isinstance(event, TextMessage):
                        if not isinstance(self.incoming_message, bytearray):
                            # convert to bytearray and append
                            self.incoming_message = bytearray(
                                (cast(str, self.incoming_message) + event.data).encode())
                        else:
                            # append to bytearray
                            self.incoming_message += event.data.encode()
                    else:
                        if not isinstance(self.incoming_message, bytearray):
                            # convert to mutable bytearray and append
                            self.incoming_message = bytearray(
                                cast(bytes, self.incoming_message) + event.data)
                        else:
                            # append to bytearray
                            self.incoming_message += event.data
                    if not event.message_finished:
                        continue
                    if isinstance(self.incoming_message, (str, bytes)):
                        # single part message
                        self.input_buffer.append(self.incoming_message)
                    elif isinstance(event, TextMessage):
                        # convert multi-part message back to text
                        self.input_buffer.append(
                            cast(bytes, self.incoming_message).decode())
                    else:
                        # convert multi-part message back to bytes
                        self.input_buffer.append(bytes(self.incoming_message))
                    self.incoming_message = None
                    self.incoming_message_len = 0
                    self.event.set()
                else:  # pragma: no cover
                    pass
            except LocalProtocolError:  # pragma: no cover
                out_data = b''
                self.event.set()
                keep_going = False
        if out_data:
            self.sock.send(out_data)
        return keep_going


#
# End of licensed code
#


def http_handshake(headers, extensions):
    """
    HTTP handshake for WebSocket connections.
    
    Notes:
    * Method must be GET
    """
    headers = {k.lower(): headers[k] for k in headers.keys() or []}
    version = headers.pop("sec-websocket-version", None)
    upgrade = headers.pop("upgrade", None)
    key = headers.pop("sec-websocket-key", None)
    headers.pop("host", None)
    connection_tokens = headers.pop("connection", None)
    del connection_tokens
    extensions_ = headers.pop("sec-websocket-extensions", None)
    headers.pop("sec-websocket-protocol", None)

    if not version:
        logging.info("Handshake failed: missing header, Sec-WebSocket-Version")
        return 400, b"", {}
    if version != WEBSOCKET_VERSION.decode("ascii"):
        logging.info(f"Handshake failed: unsupported Sec-WebSocket-Version: {version}")
        return 426, b"", {}
    if (upgrade or "").lower() != WEBSOCKET_UPGRADE.decode("ascii"):
        logging.info(f"Handshake failed: missing header, Upgrade: {WEBSOCKET_UPGRADE}")
        return 400, b"", {}
    if key is None:
        logging.info("Handshake failed: missing header, Sec-WebSocket-Key")
        return 400, b"", {}

    accept_token = base64.b64encode(
        hashlib.sha1((key + ACCEPT_GUID.decode("ascii")).encode("ascii")).digest())
    response_headers = [
        ("Upgrade", WEBSOCKET_UPGRADE.decode("ascii")),
        ("Connection", "Upgrade"),
        ("Sec-WebSocket-Accept", accept_token.decode("ascii")),
    ]

    websocket_extensions_header = ""
    for offer in (extensions_ or "").split(","):
        if not offer: continue
        name = offer.split(";", 1)[0].strip()
        for extension in extensions:
            if extension.name == name:
                accept = extension.accept(offer)
                if isinstance(accept, bool):
                    websocket_extensions_header += f", {name}"
                elif accept is not None:
                    websocket_extensions_header += f", {name}; {accept}"
    if websocket_extensions_header:
        response_headers.append(("Sec-WebSocket-Extensions", websocket_extensions_header[2:]))
    return 101, b"", response_headers



def httpserver_websocket_handler(fn) -> Callable:
    """
    A decorator function to be used with http.server.BaseHTTPRequestHandler to handle WebSocket connections.
    It is used to wrap a method that will handle the WebSocket connection.
    The method takes the websocket object as the first argument.
    """
    def websocket_handler(self, *args, **kwargs):
        extensions = [PerMessageDeflate()]
        status, setup_message, headers = http_handshake(self.headers, extensions)
        if status != 101:
            self.send_response(status)
            for key, value in headers:
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(setup_message)
            return

        sock = self.request
        # Send handshake message
        message = b'HTTP/1.1 101 \r\n'
        for key, value in headers:
            message += f'{key}: {value}\r\n'.encode()
        message += b'\r\n'
        sock.send(message)

        ws = None
        try:
            # Start the server pulling thread
            ws = Server(sock, extensions=extensions)

            # Run the user's WebSocket handler
            fn(self, ws, *args, **kwargs)
        except ConnectionClosed:
            pass
        if ws is not None:
            try:
                ws.close()
            except:  # noqa: E722
                pass

    return websocket_handler


def flask_websocket_route(app, *args, **kwargs):
    Response = app.response_class
    from flask import request, Response

    def wrap(fn):
        @app.route(*args, websocket=True, **kwargs)
        def websocket():
            extensions = [PerMessageDeflate()]
            status, setup_message, headers = http_handshake(request.headers, extensions)
            if status != 101:
                return Response(setup_message, status=status, headers=headers)
            sock, mode = _get_sock(request.environ)

            # Send handshake message
            message = b'HTTP/1.1 101 \r\n'
            for key, value in headers:
                message += f'{key}: {value}\r\n'.encode()
            message += b'\r\n'
            sock.send(message)

            ws = None
            try:
                # Start the server pulling thread
                ws = Server(sock, extensions=extensions)

                # Run the user's WebSocket handler
                fn(ws)
            except ConnectionClosed:
                pass
            if ws is not None:
                try:
                    ws.close()
                except:  # noqa: E722
                    pass

            class WebSocketResponse(Response):
                def __call__(self, *args, **kwargs):
                    del kwargs
                    if mode == 'eventlet':
                        try:
                            from eventlet.wsgi import WSGI_LOCAL
                            ALREADY_HANDLED = []
                        except ImportError:
                            from eventlet.wsgi import ALREADY_HANDLED  # type: ignore
                            WSGI_LOCAL = None

                        if hasattr(WSGI_LOCAL, 'already_handled'):
                            setattr(WSGI_LOCAL, 'already_handled', True)
                        return ALREADY_HANDLED
                    elif mode == 'gunicorn':
                        raise StopIteration()
                    elif mode == 'werkzeug':
                        # A hack to prevent Werkzeug from sending the response
                        # NOTE: This hack is very dirty and may not work in future versions of Werkzeug
                        start_response = args[1]
                        write = start_response(object(), headers)
                        try: write(b'')
                        except Exception: pass
                        return []
                    else:
                        return []

            return WebSocketResponse()
        del websocket
    return wrap


if __name__ == "__main__":
    # Testing code for websocket
    from flask import Flask
    app = Flask(__name__)

    @flask_websocket_route(app, "/websocket")
    def websocket(ws):
        while True:
            ws.send('hi')
            print(ws.receive())
    del websocket

    @app.route("/index.html")
    def index():
        return '''<html>
<script>
async function main() {
let socket = await new Promise((resolve, reject) => {
    try {
      const socket = new WebSocket("./websocket");
      // socket.binaryType = "blob";
      socket.addEventListener("open", () => {
        console.log("WebSocket connection established");
        resolve(socket);
      });
    } catch (error) {
      reject(error);
    }
});

window.socket = socket;
socket.addEventListener("close", () => {
    console.log("WebSocket connection closed");
    socket = undefined;
});
socket.addEventListener("error", (error) => {
    console.error("WebSocket error:", error);
});
socket.addEventListener("message", async (event) => {
  console.log(event.data);
    await socket.send("Hello, WebSocket!");
    console.log("Message sent");
    socket.close();
});
};
main();
</script>
</html>'''

    app.run(host="0.0.0.0", port=5010)
